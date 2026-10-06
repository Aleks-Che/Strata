// Included in the private generated ggml-backend.cpp, not the source archive.
#include "sync_runtime.h"
#include "sync_test.h"
#include "../common/expert_file.hpp"
#include "../common/device_memory.hpp"
#include "../glm5next/host_pages.hpp"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
static thread_local int hy3_sync_mode=0;
static thread_local strata_hy3_sync_stats hy3_sync_stats{};
static thread_local void * hy3_staging=nullptr;
static constexpr size_t hy3_staging_limit=16*1024*1024;
static thread_local strata_hy3_copy_observer hy3_observer=nullptr;
static thread_local void * hy3_owner=nullptr;
static thread_local const std::atomic<bool> * hy3_cancel=nullptr;
static thread_local std::unique_ptr<strata_glm::HostWorkingSetBudget> hy3_ram_budget;
static thread_local strata_hy3_test_limits hy3_test_limits;
extern "C" void strata_hy3_test_set_limits(strata_hy3_test_limits limits) {hy3_test_limits=limits;}
static const std::atomic<bool> hy3_never_cancel{false};
static void hy3_cuda(cudaError_t rc) {if(rc!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));}
void strata_hy3_sync_mode(int mode) {
    if(mode<0 || mode>2) throw std::invalid_argument("invalid Hy3 copy mode");
    hy3_sync_mode=mode;
}
void strata_hy3_sync_cancel(const std::atomic<bool> * cancel) {hy3_cancel=cancel;}
void strata_hy3_sync_observer(strata_hy3_copy_observer fn,void * owner) {hy3_observer=fn;hy3_owner=owner;}
void strata_hy3_sync_reset() {hy3_sync_stats={};hy3_sync_stats.staging_bytes=hy3_staging ? hy3_staging_limit : 0;}
strata_hy3_sync_stats strata_hy3_sync_snapshot() {return hy3_sync_stats;}
void strata_hy3_memory_check(size_t reserve) {
    size_t free=0,total=0;
#ifdef _WIN32
    static StrataGlobalMemory gpu;
    int device=-1;hy3_cuda(cudaGetDevice(&device));
    if(!gpu.sample(device,free,total)) throw std::runtime_error("Hy3 global VRAM query failed");
    if(!hy3_ram_budget) hy3_ram_budget=std::make_unique<strata_glm::HostWorkingSetBudget>();
    // Bound this process's mapped resident pages before the global 95% guard.
    // The extra 2% margin absorbs driver allocations and other-process changes.
    hy3_sync_stats.working_set_limit=hy3_ram_budget->apply(93);
    MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
    if(!GlobalMemoryStatusEx(&ram)) throw std::runtime_error("cannot query global RAM");
    hy3_sync_stats.ram_total=ram.ullTotalPhys;hy3_sync_stats.ram_free=ram.ullAvailPhys;
    if(std::min<size_t>(ram.ullAvailPhys,hy3_test_limits.ram_available)<ram.ullTotalPhys/20+hy3_staging_limit)
        throw std::runtime_error("Hy3 global RAM 95% limit reached");
#else
    throw std::runtime_error("Hy3 native-file baseline currently requires Windows");
#endif
    hy3_sync_stats.gpu_free=free;hy3_sync_stats.gpu_total=total;
    ++hy3_sync_stats.memory_samples;
    const size_t available=std::min(free,hy3_test_limits.gpu_available);
    if(available<total/20 || reserve>available-total/20) throw std::runtime_error("Hy3 global VRAM 95% limit reached");
}
void strata_hy3_sync_release() {
    if(hy3_staging) cudaFreeHost(hy3_staging);
    hy3_staging=nullptr;hy3_observer=nullptr;hy3_owner=nullptr;hy3_cancel=nullptr;hy3_sync_mode=0;
    hy3_ram_budget.reset();
    hy3_test_limits={};
}
static bool hy3_expert_weight(const ggml_tensor * t) {
    const std::string name=t->name;
    return name.rfind("blk.",0)==0 && name.size()>12 && name.compare(name.size()-12,12,"_exps.weight")==0;
}
static bool hy3_cancelled() {return hy3_cancel && hy3_cancel->load();}
static bool hy3_copy_range(ggml_backend_t backend,ggml_tensor * dst,const ggml_tensor * src,size_t offset,size_t bytes) {
    try {
        if(hy3_cancelled()) return false;
        if(!hy3_expert_weight(src) || !src->data || src->type!=dst->type || !ggml_are_same_shape(src,dst) ||
            !ggml_is_contiguous(src) || !ggml_is_contiguous(dst) || src->ne[3]!=1 ||
            offset%src->nb[2] || offset>ggml_nbytes(src) || bytes>ggml_nbytes(src)-offset ||
            offset>ggml_nbytes(dst) || bytes>ggml_nbytes(dst)-offset)
            throw std::runtime_error("invalid Hy3 selected expert range");
        const size_t tail=bytes%src->nb[2];
        if(bytes<src->nb[2] || (tail && tail!=std::min<size_t>(512,src->nb[2])))
            throw std::runtime_error("invalid Hy3 selected expert padding");
        auto file=strata_expert_file::find(src->data,ggml_nbytes(src));
        if(!file) throw std::runtime_error("unregistered Hy3 expert source mapping");
        if(!hy3_staging) {
            strata_hy3_memory_check();
            if(hy3_test_limits.fail_next_pinned_allocation) {
                hy3_test_limits.fail_next_pinned_allocation=false;
                hy3_cuda(cudaErrorMemoryAllocation);
            }
            hy3_cuda(cudaHostAlloc(&hy3_staging,hy3_staging_limit,cudaHostAllocDefault));
            hy3_sync_stats.staging_bytes=hy3_staging_limit;
        }
        // Fence scratch consumers and every upload before staging reuse.
        ggml_backend_synchronize(backend);
        strata_expert_file::Request read;
        using clock=std::chrono::steady_clock;
        const auto elapsed=[](clock::time_point start){return std::chrono::duration<double,std::milli>(clock::now()-start).count();};
        const uint64_t file_offset=file->offset_of(src->data,ggml_nbytes(src))+offset;
        ++hy3_sync_stats.ranges;
        for(size_t done=0;done<bytes;) {
            const size_t count=std::min(bytes-done,hy3_staging_limit);
            auto start=clock::now();
            if(!read.read_at(*file,file_offset+done,hy3_staging,count,hy3_cancel ? *hy3_cancel : hy3_never_cancel)) return false;
            hy3_sync_stats.source_ms+=elapsed(start);
            start=clock::now();
            ggml_backend_tensor_set_async(backend,dst,hy3_staging,offset+done,count);
            ggml_backend_synchronize(backend);
            hy3_sync_stats.h2d_ms+=elapsed(start);
            hy3_sync_stats.source_bytes+=count;hy3_sync_stats.h2d_bytes+=count;
            ++hy3_sync_stats.chunks;done+=count;
        }
        if(hy3_observer) hy3_observer(hy3_owner,backend,dst,src,offset,bytes);
        return true;
    } catch(const std::exception & e) {GGML_LOG_ERROR("STRATA_HY3 selected-copy: %s\n",e.what());return false;}
}
