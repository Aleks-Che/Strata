#pragma once
#include "../common/device_memory.hpp"
#include "../common/vram_policy.hpp"
#include "../common/expert_frequency.hpp"
#include <chrono>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace step35 {
inline void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("Step CUDA: ")+cudaGetErrorString(error));
}
struct MemorySample { size_t gpu_free=0, gpu_total=0, ram_free=0, ram_total=0; };
inline MemorySample memory_sample() {
    MemorySample out;
#ifdef _WIN32
    static StrataGlobalMemory gpu;
    int device=-1; cuda_check(cudaGetDevice(&device));
    if (!gpu.sample(device,out.gpu_free,out.gpu_total)) throw std::runtime_error("Step global VRAM sample unavailable");
    MEMORYSTATUSEX ram{}; ram.dwLength=sizeof(ram);
    if (!GlobalMemoryStatusEx(&ram)) throw std::runtime_error("Step RAM sample unavailable");
    out.ram_total=size_t(ram.ullTotalPhys); out.ram_free=size_t(ram.ullAvailPhys);
#else
    cuda_check(cudaMemGetInfo(&out.gpu_free,&out.gpu_total));
    std::ifstream input("/proc/meminfo"); std::string line;
    while (std::getline(input,line)) {
        std::istringstream in(line); std::string key; size_t kib=0; in>>key>>kib;
        if (key=="MemTotal:") out.ram_total=kib*1024;
        if (key=="MemAvailable:") out.ram_free=kib*1024;
    }
#endif
    if (!out.gpu_total || out.gpu_free>out.gpu_total || !out.ram_total || out.ram_free>out.ram_total)
        throw std::runtime_error("invalid Step memory sample");
    return out;
}
// Keys are tied to one immutable live model, never persisted or reused by path.
// Registry IDs identify full tensor metadata (shard/offset/type/shape) below.
struct MatrixKey {
    uint64_t generation=0; uint32_t tensor=0, expert=0;
    auto fields() const { return std::tie(generation,tensor,expert); }
    bool operator==(const MatrixKey & b) const { return fields()==b.fields(); }
    bool operator<(const MatrixKey & b) const { return fields()<b.fields(); }
};
struct MatrixHash {
    size_t operator()(const MatrixKey & k) const {
        return std::hash<uint64_t>{}(k.generation) ^ (size_t(k.tensor)<<16) ^ k.expert;
    }
};
// Synchronous cache: every caller must complete D2D reads before another lookup,
// trim or admission. There are no outstanding leases/retired allocations here.
// Async delivery must introduce event-protected leases before relaxing this rule.
class ExpertCache {
public:
    using Probe=std::function<MemorySample()>;
    using Allocate=std::function<cudaError_t(void **,size_t)>;
    struct Counters { uint64_t hits=0,misses=0,evictions=0,bypasses=0,oom=0,rejected=0,samples=0; };
private:
    struct Entry { void * data; size_t bytes,allocated; std::list<MatrixKey>::iterator order; size_t pins=0; };
    std::map<MatrixKey,Entry> entries;
    std::list<MatrixKey> order;
    StrataExpertFrequencyHistory<MatrixKey,MatrixHash> history{65536};
    size_t cap=0,limit=0,resident=0,growth=0;
    Probe probe;
    Allocate allocate;
    MemorySample sample{};
    Counters counts{};
    int device=-1;
    void check_device() const {
        int current=-1; cuda_check(cudaGetDevice(&current));
        if (current!=device) throw std::runtime_error("Step cache device changed");
    }
    void erase(std::map<MatrixKey,Entry>::iterator it) {
        cuda_check(cudaFree(it->second.data)); resident-=it->second.allocated;
        order.erase(it->second.order); entries.erase(it); ++counts.evictions;
    }
public:
    // Protect future hits while an ordered pipeline plan admits other misses.
    // These are residency pins, not asynchronous CUDA consumer leases.
    class PlanPins {
        friend class ExpertCache;
        ExpertCache * owner;
        std::vector<MatrixKey> keys;
        PlanPins(ExpertCache * c,std::vector<MatrixKey> k):owner(c),keys(std::move(k)) {
            for (const auto & key:keys) ++owner->entries.find(key)->second.pins;
        }
    public:
        PlanPins(const PlanPins &)=delete;
        ~PlanPins() {for (const auto & key:keys) --owner->entries.find(key)->second.pins;}
    };
    bool contains(const MatrixKey & key,size_t bytes) const {
        const auto it=entries.find(key);
        if (it!=entries.end() && it->second.bytes!=bytes) throw std::runtime_error("Step planned cache size changed");
        return it!=entries.end();
    }
    std::unique_ptr<PlanPins> protect(const std::vector<MatrixKey> & keys) {
        check_device();
        std::vector<MatrixKey> hits;hits.reserve(keys.size());
        for (const auto & key:keys) if (entries.count(key)) hits.push_back(key);
        return std::unique_ptr<PlanPins>(new PlanPins(this,std::move(hits)));
    }
    explicit ExpertCache(size_t bytes, Probe reader=memory_sample,
        Allocate allocator=[](void ** p,size_t n){return cudaMalloc(p,n);})
        :cap(bytes),probe(std::move(reader)),allocate(std::move(allocator)) { cuda_check(cudaGetDevice(&device)); }
    ExpertCache(const ExpertCache &)=delete;
    ExpertCache &operator=(const ExpertCache &)=delete;
    ~ExpertCache() {
        int previous=device; cudaGetDevice(&previous); cudaSetDevice(device);
        for (auto & pair:entries) cudaFree(pair.second.data);
        cudaSetDevice(previous);
    }
    void refresh() {
        check_device(); ++counts.samples;
        // Fail closed on unavailable global readings, including WDDM/NVML.
        limit=0; sample=probe();
        if (!sample.gpu_total || sample.gpu_free>sample.gpu_total || !sample.ram_total || sample.ram_free>sample.ram_total)
            throw std::runtime_error("invalid Step memory sample");
        if (resident>sample.gpu_total-sample.gpu_free) {
            trim(0);
            throw std::runtime_error("Step global VRAM sample cannot account for resident cache");
        }
        StrataVramPolicy policy;
        // Leave 5% plus 256 MiB for sampling lag/driver allocation overhead.
        policy.reserve_mib=(sample.gpu_total/20 + (1<<20)-1)/(1<<20)+256;
        const size_t cached_before=resident;
        limit=size_t(std::min<uint64_t>(cap,policy.byte_limit(sample.gpu_free,sample.gpu_total,cached_before)));
        trim(limit); growth=0;
        if (sample.ram_free < sample.ram_total/20 + (64<<20)) {
            trim(0);
            throw std::runtime_error("Step RAM budget95 admission refused");
        }
        const size_t used=sample.gpu_total-sample.gpu_free;
        const size_t fixed=used-std::min(used,cached_before);
        if (fixed>sample.gpu_total-sample.gpu_total/20)
            throw std::runtime_error("Step fixed/external VRAM exceeds budget95");
    }
    void trim(size_t bytes) {
        check_device(); limit=std::min(limit,bytes);
        for (auto key=order.begin();resident>limit && key!=order.end();) {
            auto it=entries.find(*key++);
            if (!it->second.pins) erase(it);
        }
    }
    void * get(const MatrixKey & key,size_t bytes) {
        check_device(); history.record(key);
        const auto it=entries.find(key);
        if (it==entries.end()) {++counts.misses;return nullptr;}
        if (it->second.bytes!=bytes) throw std::runtime_error("Step cache key size changed");
        ++counts.hits; order.splice(order.end(),order,it->second.order); return it->second.data;
    }
    void * admit(const MatrixKey & key,size_t bytes) {
        check_device();
        if (growth>=64*1024*1024) refresh();
        const size_t charged=(bytes+65535)/65536*65536;
        if (!bytes || charged<bytes || charged>limit) {++counts.bypasses;return nullptr;}
        while (resident>limit-charged) {
            auto victim=entries.end();
            // Bounded scan of the oldest entries; shared decaying frequency
            // history avoids admitting every one-off prefill matrix.
            int scanned=0;
            for (auto it=order.begin();it!=order.end() && scanned<32;++it) {
                auto candidate=entries.find(*it);
                if (candidate->second.pins) continue;
                ++scanned;
                if (victim==entries.end() || history.score(candidate->first)<history.score(victim->first)) victim=candidate;
            }
            if (victim==entries.end()) {++counts.bypasses;return nullptr;}
            if (history.score(key)<history.score(victim->first)) {++counts.rejected;return nullptr;}
            erase(victim);
        }
        void * data=nullptr;
        const auto error=allocate(&data,charged);
        if (error==cudaErrorMemoryAllocation) {cudaGetLastError();++counts.oom;++counts.bypasses;return nullptr;}
        cuda_check(error);
        try {
            order.push_back(key);
            try {
                if (!entries.emplace(key,Entry{data,bytes,charged,std::prev(order.end())}).second)
                    throw std::runtime_error("duplicate Step cache admission");
            } catch (...) {order.pop_back();throw;}
        } catch (...) {cudaFree(data);throw;}
        resident+=charged; growth+=charged; return data;
    }
    size_t resident_bytes() const {return resident;}
    size_t budget() const {return limit;}
    size_t size() const {return entries.size();}
    MemorySample memory() const {return sample;}
    Counters counters() const {return counts;}
    void reset_counters() {counts={};}
};
} // namespace step35
