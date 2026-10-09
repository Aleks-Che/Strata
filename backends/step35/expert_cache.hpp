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
// Opt-in host wall time, including waits/preemption. Nested scopes are inclusive
// and must not be added together. Disabled scopes do not read the clock.
class CpuTimer {
    double * total;
    std::chrono::steady_clock::time_point start;
public:
    explicit CpuTimer(double * value):total(value) {if (total) start=std::chrono::steady_clock::now();}
    ~CpuTimer() {if (total) *total+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
};
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
// Callers normally complete D2D reads before another lookup/trim/admission.
// The tensor-batched pipeline may queue copies only while PlanPins protect ALL
// pending cache sources and destinations. It must drain its CUDA stream before
// releasing any of those pins, including on exceptions. No retired pool exists.
class ExpertCache {
public:
    using Probe=std::function<MemorySample()>;
    using Allocate=std::function<cudaError_t(void **,size_t)>;
    using Release=std::function<cudaError_t(void *)>;
    using Backing=std::function<size_t()>;
    using BackingGroup=std::function<uintptr_t(const void *)>;
    struct Counters {
        uint64_t hits=0,misses=0,evictions=0,bypasses=0,oom=0,rejected=0,samples=0;
        uint64_t allocations=0,reuses=0;
        uint64_t victim_candidates=0;
        uint64_t pressure_trims=0,pressure_groups=0,pressure_evicted_bytes=0,pressure_released_bytes=0;
        double get_ms=0,admit_ms=0,refresh_ms=0,probe_ms=0,protect_ms=0,trim_ms=0;
        double victim_ms=0,allocate_ms=0,free_ms=0;
    };
private:
    struct Entry;
    // std::map nodes keep their addresses until erased. Remove the order node
    // before its entry on every eviction, including allocation reuse.
    struct OrderItem {MatrixKey key; Entry * entry;};
    struct Entry { void * data; size_t bytes,allocated; std::list<OrderItem>::iterator order; size_t pins=0; };
    std::map<MatrixKey,Entry> entries;
    std::list<OrderItem> order;
    StrataExpertFrequencyHistory<MatrixKey,MatrixHash> history;
    size_t cap=0,limit=0,resident=0,growth=0;
    bool reuse_allocations=false;
    bool match_size=false;
    bool profile=false;
    bool fast_scan=false;
    Probe probe;
    Allocate allocate;
    Release release;
    Backing backing;
    BackingGroup backing_group;
    MemorySample sample{};
    Counters counts{};
    int device=-1;
    void check_device() const {
        int current=-1; cuda_check(cudaGetDevice(&current));
        if (current!=device) throw std::runtime_error("Step cache device changed");
    }
    void erase(std::map<MatrixKey,Entry>::iterator it) {
        { CpuTimer timer(profile?&counts.free_ms:nullptr); cuda_check(release(it->second.data)); }
        resident-=it->second.allocated;
        order.erase(it->second.order); entries.erase(it); ++counts.evictions;
    }
    void trim_backing() {
        if(backing()<=limit)return;
        CpuTimer timer(profile?&counts.trim_ms:nullptr);
        ++counts.pressure_trims;
        struct Group {size_t bytes=0,newest=0;bool pinned=false;std::vector<MatrixKey> keys;};
        std::map<uintptr_t,Group> groups;
        size_t age=0;
        // Only build this index under physical pressure. Inspect every pin
        // before releasing anything: one pending user protects its whole slab.
        for(const auto &item:order) {
            const auto id=backing_group(item.entry->data);
            if(!id)throw std::runtime_error("Step cache invalid backing group");
            auto &g=groups[id];g.bytes+=item.entry->allocated;g.newest=age++;
            g.pinned|=item.entry->pins!=0;g.keys.push_back(item.key);
        }
        std::vector<Group *> victims;victims.reserve(groups.size());
        for(auto &pair:groups)if(!pair.second.pinned)victims.push_back(&pair.second);
        // Sacrifice the least live payload first. Equal-sized groups prefer
        // the one whose most recent access is oldest. Normal replacement and
        // frequency training are unchanged; this only returns physical blocks.
        std::sort(victims.begin(),victims.end(),[](const Group *a,const Group *b) {
            return std::tie(a->bytes,a->newest)<std::tie(b->bytes,b->newest);
        });
        for(const auto *g:victims) {
            const auto before=backing();if(before<=limit)break;
            for(const auto &key:g->keys)erase(entries.find(key));
            const auto after=backing();
            if(after>=before)throw std::runtime_error("Step cache backing group did not release memory");
            ++counts.pressure_groups;counts.pressure_evicted_bytes+=g->bytes;
            counts.pressure_released_bytes+=before-after;
        }
        if(backing()>limit)
            throw std::runtime_error("Step pinned cache cannot shrink under memory pressure; drain plan first");
    }
    void *matching_reuse(const MatrixKey &key,size_t charged) {
        auto victim=entries.end();unsigned score=0;size_t scanned=0;
        for(const auto &item:order) {
            if(item.entry->pins)continue;
            ++scanned;
            if(item.entry->allocated==charged) {
                const auto value=history.score(item.key);
                if(victim==entries.end() || value<score) {victim=entries.find(item.key);score=value;}
            }
            if(scanned==64)break;
        }
        if(profile)counts.victim_candidates+=scanned;
        if(victim==entries.end() || history.score(key)<score)return nullptr;
        void *data=victim->second.data;
        resident-=charged;order.erase(victim->second.order);entries.erase(victim);
        ++counts.evictions;++counts.reuses;return data;
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
        CpuTimer timer(profile?&counts.protect_ms:nullptr);
        check_device();
        std::vector<MatrixKey> hits;hits.reserve(keys.size());
        for (const auto & key:keys) if (entries.count(key)) hits.push_back(key);
        return std::unique_ptr<PlanPins>(new PlanPins(this,std::move(hits)));
    }
    explicit ExpertCache(size_t bytes, Probe reader=memory_sample,
        Allocate allocator=[](void ** p,size_t n){return cudaMalloc(p,n);},
        Release releaser=[](void *p){return cudaFree(p);},uint64_t decay_period=65536)
        :history(decay_period),cap(bytes),probe(std::move(reader)),allocate(std::move(allocator)),release(std::move(releaser)) { cuda_check(cudaGetDevice(&device)); }
    ExpertCache(const ExpertCache &)=delete;
    ExpertCache &operator=(const ExpertCache &)=delete;
    ~ExpertCache() {
        int previous=device; cudaGetDevice(&previous); cudaSetDevice(device);
        for (auto & pair:entries) release(pair.second.data);
        cudaSetDevice(previous);
    }
    void refresh() {
        CpuTimer timer(profile?&counts.refresh_ms:nullptr);
        check_device(); ++counts.samples;
        // Fail closed on unavailable global readings, including WDDM/NVML.
        limit=0;
        { CpuTimer probe_timer(profile?&counts.probe_ms:nullptr); sample=probe(); }
        if (!sample.gpu_total || sample.gpu_free>sample.gpu_total || !sample.ram_total || sample.ram_free>sample.ram_total)
            throw std::runtime_error("invalid Step memory sample");
        const size_t allocated=backing?backing():resident;
        if (allocated<resident || allocated>sample.gpu_total-sample.gpu_free) {
            trim(0);
            throw std::runtime_error("Step global VRAM sample cannot account for resident cache");
        }
        StrataVramPolicy policy;
        // Leave 5% plus 256 MiB for sampling lag/driver allocation overhead.
        policy.reserve_mib=(sample.gpu_total/20 + (1<<20)-1)/(1<<20)+256;
        // Arena holes are already allocated to this cache and can hold future
        // matrices without consuming new physical VRAM. Do not classify them
        // as external memory and repeatedly shrink useful residency.
        const size_t cached_before=allocated;
        limit=size_t(std::min<uint64_t>(cap,policy.byte_limit(sample.gpu_free,sample.gpu_total,cached_before)));
        if(backing_group)trim_backing();else trim(limit);
        growth=0;
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
        CpuTimer timer(profile?&counts.trim_ms:nullptr);
        check_device(); limit=std::min(limit,bytes);
        for (auto key=order.begin();resident>limit && key!=order.end();) {
            auto it=entries.find((key++)->key);
            if (!it->second.pins) erase(it);
        }
    }
    // Prefill can serve existing weights without teaching a decode-only cache
    // that a one-off prompt scan is popular. Default preserves other backends.
    void * get(const MatrixKey & key,size_t bytes,bool train=true) {
        CpuTimer timer(profile?&counts.get_ms:nullptr);
        check_device(); if(train)history.record(key);
        const auto it=entries.find(key);
        if (it==entries.end()) {++counts.misses;return nullptr;}
        if (it->second.bytes!=bytes) throw std::runtime_error("Step cache key size changed");
        ++counts.hits; if(train)order.splice(order.end(),order,it->second.order); return it->second.data;
    }
    void * admit(const MatrixKey & key,size_t bytes) {
        CpuTimer timer(profile?&counts.admit_ms:nullptr);
        check_device();
        if (entries.count(key)) throw std::runtime_error("duplicate Step cache admission");
        if (growth>=64*1024*1024) refresh();
        const size_t charged=(bytes+65535)/65536*65536;
        if (!bytes || charged<bytes || charged>limit) {++counts.bypasses;return nullptr;}
        void * data=nullptr;
        // Packed arenas can have free bytes in another size class. Prefer a
        // same-size, unpinned, no-hotter victim near the LRU tail (opt-in).
        if(match_size && reuse_allocations && resident>limit-charged && resident<=limit)
            data=matching_reuse(key,charged);
        while (!data && resident>limit-charged) {
            auto victim=entries.end();
            // Bounded scan of the oldest entries; shared decaying frequency
            // history avoids admitting every one-off prefill matrix.
            int scanned=0;
            {
            CpuTimer victim_timer(profile?&counts.victim_ms:nullptr);
            if (fast_scan) {
                const MatrixKey * best=nullptr;unsigned best_score=0;
                for (const auto & item:order) {
                    if (item.entry->pins) continue;
                    ++scanned;
                    const unsigned score=history.score(item.key);
                    if (!best || score<best_score) {best=&item.key;best_score=score;}
                    // Scores cannot be negative; the first minimum wins ties.
                    // No history.record() occurs during this scan.
                    if (!best_score || scanned==32) break;
                }
                if (best) victim=entries.find(*best);
            } else {
                for (auto it=order.begin();it!=order.end() && scanned<32;++it) {
                    auto candidate=entries.find(it->key);
                    if (candidate->second.pins) continue;
                    ++scanned;
                    if (victim==entries.end() || history.score(candidate->first)<history.score(victim->first)) victim=candidate;
                }
            }
            if (profile) counts.victim_candidates+=scanned;
            }
            if (victim==entries.end()) {++counts.bypasses;return nullptr;}
            if (history.score(key)<history.score(victim->first)) {++counts.rejected;return nullptr;}
            // Reads are complete by the cache contract; plan-pinned hits were
            // excluded above. Reuse only an equal-sized allocation when this
            // single replacement fits the current budget. No retained pool and
            // no extra residency. The caller must fill all bytes before lookup.
            if (reuse_allocations && victim->second.allocated==charged && resident<=limit) {
                data=victim->second.data;
                resident-=charged;
                order.erase(victim->second.order); entries.erase(victim);
                ++counts.evictions; ++counts.reuses;
                break;
            }
            erase(victim);
        }
        if (!data) {
            ++counts.allocations;
            CpuTimer allocation_timer(profile?&counts.allocate_ms:nullptr);
            const auto error=allocate(&data,charged);
            if (error==cudaErrorMemoryAllocation) {
                cudaGetLastError();++counts.oom;
                // A new slab may not fit although the logical byte cap has
                // room. Reuse a compatible victim instead of freezing admission.
                if(match_size && reuse_allocations && resident<=limit)data=matching_reuse(key,charged);
                if(!data) {++counts.bypasses;return nullptr;}
            } else cuda_check(error);
        }
        try {
            order.push_back({key,nullptr});
            try {
                auto inserted=entries.emplace(key,Entry{data,bytes,charged,std::prev(order.end())});
                if (!inserted.second)
                    throw std::runtime_error("duplicate Step cache admission");
                order.back().entry=&inserted.first->second;
            } catch (...) {order.pop_back();throw;}
        } catch (...) {release(data);throw;}
        resident+=charged; growth+=charged; return data;
    }
    size_t resident_bytes() const {return resident;}
    // Read-only accounting; does not change frequency, recency or pins.
    size_t tensor_bytes(uint64_t generation,uint32_t tensor) const {
        size_t bytes=0;
        for(auto it=entries.lower_bound({generation,tensor,0});it!=entries.end() &&
            it->first.generation==generation && it->first.tensor==tensor;++it) bytes+=it->second.allocated;
        return bytes;
    }
    size_t budget() const {return limit;}
    size_t size() const {return entries.size();}
    MemorySample memory() const {return sample;}
    Counters counters() const {return counts;}
    void reset_counters() {counts={};}
    void set_reuse_allocations(bool enabled) {check_device();reuse_allocations=enabled;}
    void set_match_size(bool enabled) {check_device();match_size=enabled;}
    void set_backing_bytes(Backing reader) {check_device();backing=std::move(reader);}
    // Opt-in for allocators whose release returns backing only after all slots
    // in a group are gone. The allocator must be exclusively owned by this cache.
    // Applies to BOTH explicit refresh and the refresh inside admit().
    void set_backing_groups(BackingGroup reader) {
        check_device();
        if(reader && !backing)throw std::runtime_error("Step backing groups require byte accounting");
        backing_group=std::move(reader);
    }
    void set_profile(bool enabled) {check_device();profile=enabled;}
    void set_fast_scan(bool enabled) {check_device();fast_scan=enabled;}
};
} // namespace step35
