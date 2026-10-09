// Opt-in HTTP lifecycle diagnostic. Own allocations only; no model execution.
#include "../common/device_memory.hpp"
#include "../glm5next/host_pages.hpp"
#include "nlohmann/json.hpp"
#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>

using json=nlohmann::ordered_json;
static constexpr uint64_t MiB=1ull<<20;
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
static void cuda_check(cudaError_t rc) {if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}

struct Holder {
    StrataGlobalMemory device;
    strata_glm::HostWorkingSetBudget host_budget;
    std::vector<void *> gpu;
    std::vector<void *> ram_private;
    HANDLE file=INVALID_HANDLE_VALUE,mapping=nullptr;
    void *view=nullptr;
    uint64_t gpu_bytes=0,ram_touched=0,length=0,ram_private_bytes=0;
    unsigned checksum=0;
    ~Holder() {release();}
    void release() {
        for(auto p:gpu)cudaFree(p);
        gpu.clear();gpu_bytes=0;
        for(auto p:ram_private)VirtualFree(p,0,MEM_RELEASE);
        ram_private.clear();ram_private_bytes=0;
        if(view)UnmapViewOfFile(view);
        if(mapping)CloseHandle(mapping);
        if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);
        view=nullptr;mapping=nullptr;file=INVALID_HANDLE_VALUE;ram_touched=0;
    }
    json sample(const char *phase) {
        size_t free=0,total=0;
        require(device.sample(0,free,total),"global NVML sample unavailable");
        MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
        PROCESS_MEMORY_COUNTERS_EX proc{};proc.cb=sizeof(proc);
        require(GlobalMemoryStatusEx(&ram)!=0,"RAM sample unavailable");
        require(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&proc),sizeof(proc))!=0,"process sample unavailable");
        return {{"phase",phase},{"gpu_free",free},{"gpu_total",total},{"ram_free",ram.ullAvailPhys},{"ram_total",ram.ullTotalPhys},
                {"gpu_bytes",gpu_bytes},{"ram_touched",ram_touched},{"ram_private_bytes",ram_private_bytes},{"process_working_set",proc.WorkingSetSize},
                {"process_private",proc.PrivateUsage},{"checksum",checksum}};
    }
    void guard(const json &m) {
        require(m["gpu_free"].get<uint64_t>()>=m["gpu_total"].get<uint64_t>()/20,"global VRAM95 guard");
        require(m["ram_free"].get<uint64_t>()>=m["ram_total"].get<uint64_t>()/20,"global RAM95 guard");
    }
    void allocate_gpu() {
        require(gpu.empty(),"GPU pressure already active");
        constexpr uint64_t chunk=32*MiB;
        // Stay below 94.5%; the engine trims at 95% minus its 256 MiB reserve.
        // Never replace more than 12 GiB, even if another process frees VRAM.
        while(gpu_bytes<(12ull<<30)) {
            const auto m=sample("allocating_gpu");guard(m);
            if(m["gpu_free"].get<uint64_t>() < m["gpu_total"].get<uint64_t>()*55/1000+chunk)break;
            void *p=nullptr;cuda_check(cudaMalloc(&p,chunk));gpu.push_back(p);gpu_bytes+=chunk;
            cuda_check(cudaMemset(p,0x6B,chunk));cuda_check(cudaDeviceSynchronize());
            guard(sample("allocating_gpu"));
        }
    }
    void allocate_ram(const std::filesystem::path &path) {
        require(!view,"RAM pressure already active");
        host_budget.apply(90); // Own maximum only; minimum and machine settings unchanged.
        file=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
                         nullptr,OPEN_EXISTING,FILE_FLAG_RANDOM_ACCESS,nullptr);
        require(file!=INVALID_HANDLE_VALUE,"cannot open read-only pressure source");
        LARGE_INTEGER size{};require(GetFileSizeEx(file,&size)!=0 && size.QuadPart>0,"invalid pressure source size");
        length=uint64_t(size.QuadPart);
        mapping=CreateFileMappingW(file,nullptr,PAGE_READONLY,0,0,nullptr);require(mapping!=nullptr,"cannot map pressure source");
        view=MapViewOfFile(mapping,FILE_MAP_READ,0,0,0);require(view!=nullptr,"cannot view pressure source");
        constexpr uint64_t chunk=64*MiB;
        for(uint64_t offset=0;offset<length;offset+=chunk) {
            const auto m=sample("allocating_ram");guard(m);
            if(m["ram_free"].get<uint64_t>()<m["ram_total"].get<uint64_t>()*12/100+chunk)break;
            const auto count=std::min(chunk,length-offset);
            const auto *p=static_cast<const volatile unsigned char *>(view)+offset;
            for(uint64_t i=0;i<count;i+=4096)checksum^=p[i];
            ram_touched+=count;
            if(ram_touched%(4ull<<30)==0)std::cout<<sample("progress").dump()<<'\n'<<std::flush;
        }
        guard(sample("allocated_ram"));
    }
    void topup_ram() {
        require(view!=nullptr,"RAM94 requires read-only RAM pressure first");
        // Raise only our old 90% working-set ceiling so it does not force
        // mapped-page replacement before the stricter 94.2% allocation target.
        host_budget.apply(95);
        constexpr uint64_t chunk=32*MiB;
        // Explicit opt-in for KV admission tests. Own touched private pages,
        // at most 16 GiB across repeated topups; leave >=5.8% physical RAM.
        // Reclaimed mapped pages can prevent reaching the target: report it,
        // never increase the bound or change machine/other-process settings.
        while(ram_private_bytes<(16ull<<30)) {
            const auto m=sample("allocating_private_ram");guard(m);
            MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
            require(GlobalMemoryStatusEx(&ram)!=0,"RAM commit sample unavailable");
            if(m["ram_free"].get<uint64_t>()<m["ram_total"].get<uint64_t>()*58/1000+chunk ||
               ram.ullAvailPageFile<ram.ullTotalPhys/20+chunk+(256*MiB))break;
            void *p=VirtualAlloc(nullptr,size_t(chunk),MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
            require(p!=nullptr,"cannot allocate owned private RAM pressure");
            try {ram_private.push_back(p);} catch(...) {VirtualFree(p,0,MEM_RELEASE);throw;}
            ram_private_bytes+=chunk;std::memset(p,0x4D,size_t(chunk));
            guard(sample("allocating_private_ram"));
        }
    }
};

int main(int argc,char **argv) {
    try {
        require(argc==2,"usage: pressure-holder READ_ONLY_GGUF");
        Holder holder;cuda_check(cudaSetDevice(0));cuda_check(cudaFree(nullptr));
        auto emit=[&](const char *phase){auto m=holder.sample(phase);holder.guard(m);std::cout<<m.dump()<<'\n'<<std::flush;};
        emit("ready");std::string command;
        while(std::getline(std::cin,command)) {
            if(command=="QUIT")break;
            if(command=="GPU") {holder.allocate_gpu();emit("gpu");}
            else if(command=="RAM") {holder.allocate_ram(std::filesystem::u8path(argv[1]));emit("ram");}
            else if(command=="RAM94") {holder.topup_ram();emit("ram94");}
            else if(command=="FREE") {holder.release();emit("freed");}
            else if(command=="SAMPLE")emit("sample");
            else throw std::runtime_error("invalid pressure command");
        }
        holder.release();emit("closed");return 0;
    } catch(const std::exception &error) {
        std::cout<<json({{"phase","error"},{"error",error.what()}}).dump()<<'\n'<<std::flush;return 1;
    }
}
