#pragma once
#include "expert_key.hpp"
#include "../common/expert_frequency.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace strata_glm {
// Single host owner, one CUDA device. A lease protects a plan's matrices from
// eviction. Queue all consumers on its stream before release/destruction.
// A miss can bypass when all eviction candidates are leased or still in flight.
class ExpertCache {
public:
    // Test seam: successful allocations must be cudaFree-compatible. On failure
    // the callback must return a CUDA error without allocating or enqueueing work.
    using Allocate=std::function<cudaError_t(void **,size_t)>;
private:
    static void check(cudaError_t e) {
        if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    }
    struct AllocationFailure:std::runtime_error {
        cudaError_t code;
        explicit AllocationFailure(cudaError_t e):std::runtime_error(cudaGetErrorString(e)),code(e) {}
    };
    struct Accounting {size_t bytes=0;int device;explicit Accounting(int d):device(d) {}};
    struct Entry {
        std::shared_ptr<Accounting> accounting;
        std::shared_ptr<const void> source;
        void *data=nullptr;
        size_t bytes;
        cudaEvent_t ready=nullptr;
        std::vector<cudaEvent_t> consumers;
        Entry(std::shared_ptr<Accounting> a,size_t n,std::shared_ptr<const void> owner,const Allocate &allocate)
            :accounting(std::move(a)),source(std::move(owner)),bytes(n) {
            auto allocation=allocate(&data,bytes);
            if(allocation!=cudaSuccess)throw AllocationFailure(allocation);
            auto status=cudaEventCreateWithFlags(&ready,cudaEventDisableTiming);
            if(status!=cudaSuccess) {cudaFree(data);check(status);}
            accounting->bytes+=bytes;
        }
        ~Entry() {
            int previous=accounting->device;
            cudaGetDevice(&previous);cudaSetDevice(accounting->device);
            cudaEventSynchronize(ready);
            for(auto e:consumers) {cudaEventSynchronize(e);cudaEventDestroy(e);}
            cudaEventDestroy(ready);cudaFree(data);
            accounting->bytes-=bytes;
            cudaSetDevice(previous);
        }
        static bool complete(cudaEvent_t event) {
            auto result=cudaEventQuery(event);
            if(result==cudaErrorNotReady)return false;
            check(result);return true;
        }
        bool idle() {
            if(!complete(ready))return false;
            reap();return consumers.empty();
        }
        void reap() {
            for(auto it=consumers.begin();it!=consumers.end();) {
                if(complete(*it)) {check(cudaEventDestroy(*it));it=consumers.erase(it);}
                else ++it;
            }
        }
    };
public:
    // Pin all resident matrices before processing any misses. A pin is not a
    // consumer lease: it exposes no pointer and records no CUDA work. Obtain a
    // normal get() lease for each actual consumer and retain newly loaded leases
    // for the remainder of the plan. Invalidation may retire pinned entries, but
    // their allocations remain charged until both pins and leases are released.
    class PlanPins {
        friend class ExpertCache;
        std::vector<std::shared_ptr<Entry>> entries;
        explicit PlanPins(std::vector<std::shared_ptr<Entry>> values):entries(std::move(values)) {}
    public:
        PlanPins(const PlanPins&)=delete;
        PlanPins &operator=(const PlanPins&)=delete;
        PlanPins(PlanPins&&)=default;
        PlanPins &operator=(PlanPins&&)=delete;
        size_t size() const {return entries.size();}
        void release() {entries.clear();}
    };
    class Lease {
        friend class ExpertCache;
        std::shared_ptr<Entry> entry;
        cudaStream_t stream=nullptr;
        Lease(std::shared_ptr<Entry> e,cudaStream_t s):entry(std::move(e)),stream(s) {}
    public:
        Lease()=default;
        Lease(const Lease&)=delete;
        Lease &operator=(const Lease&)=delete;
        Lease(Lease &&other) noexcept:entry(std::move(other.entry)),stream(other.stream) {}
        Lease &operator=(Lease&&)=delete;
        explicit operator bool() const {return bool(entry);}
        void *data() const {return entry?entry->data:nullptr;}
        void release() {
            if(!entry)return;
            entry->reap();
            cudaEvent_t event=nullptr;
            check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
            try {
                check(cudaEventRecord(event,stream));
                entry->consumers.push_back(event);
            }catch(...) {
                cudaStreamSynchronize(stream);cudaEventDestroy(event);throw;
            }
            entry.reset();
        }
        ~Lease() {
            if(!entry)return;
            try {release();}catch(...) {cudaStreamSynchronize(stream);entry.reset();}
        }
    };
    struct Admission {
        bool frequency=false;
        uint64_t decay_period=4096;
        size_t max_keys=131072;
    };
    struct Counters {
        uint64_t hits=0,misses=0,bypasses=0,evictions=0,invalidations=0;
        uint64_t admissions=0,admission_rejects=0,paused_bypasses=0;
        uint64_t allocation_bypasses=0;
    };
private:
    struct Cached {std::shared_ptr<Entry> entry;uint64_t used;};
    std::map<ExpertKey,Cached> entries;
    std::shared_ptr<Accounting> accounting;
    size_t budget;
    Allocate allocate;
    bool admission_enabled=true;
    uint64_t clock=0;
    Counters counts;
    std::unique_ptr<StrataExpertFrequencyHistory<ExpertKey,ExpertKeyHash>> frequency;
    void device_check() const {
        int device=-1;check(cudaGetDevice(&device));
        if(device!=accounting->device)throw std::runtime_error("GLM cache used on another CUDA device");
    }
    bool room(size_t wanted,const ExpertKey *candidate=nullptr) {
        if(candidate && frequency && (accounting->bytes>budget || wanted>budget-accounting->bytes)) {
            // Plan all victims first. A mixed-size candidate must not evict a
            // cold entry and then fail admission against the next, hotter one.
            std::vector<decltype(entries.begin())> victims;
            for(auto it=entries.begin();it!=entries.end();++it)
                if(it->second.entry.use_count()==1 && it->second.entry->idle())victims.push_back(it);
            std::sort(victims.begin(),victims.end(),[](auto a,auto b){return a->second.used<b->second.used;});
            size_t remaining=accounting->bytes,take=0;
            const auto score=frequency->score(*candidate);
            while(remaining>budget || wanted>budget-remaining) {
                if(take==victims.size())return false;
                auto victim=victims[take++];
                if(score<frequency->score(victim->first)) {++counts.admission_rejects;return false;}
                remaining-=victim->second.entry->bytes;
            }
            for(size_t i=0;i<take;++i) {entries.erase(victims[i]);++counts.evictions;}
            return true;
        }
        while(accounting->bytes>budget || wanted>budget-accounting->bytes) {
            auto victim=entries.end();
            for(auto it=entries.begin();it!=entries.end();++it) {
                if(it->second.entry.use_count()!=1 || !it->second.entry->idle())continue;
                if(victim==entries.end() || it->second.used<victim->second.used)victim=it;
            }
            if(victim==entries.end())return false;
            entries.erase(victim);++counts.evictions;
        }
        return true;
    }
public:
    explicit ExpertCache(size_t limit):ExpertCache(limit,Admission{}) {}
    ExpertCache(size_t limit,Admission admission,Allocate allocator=[](void **ptr,size_t n){return cudaMalloc(ptr,n);})
        :budget(limit),allocate(std::move(allocator)) {
        if(!allocate)throw std::invalid_argument("GLM cache requires an allocator");
        int device=-1;check(cudaGetDevice(&device));accounting=std::make_shared<Accounting>(device);
        if(admission.frequency)
            frequency=std::make_unique<StrataExpertFrequencyHistory<ExpertKey,ExpertKeyHash>>(
                admission.decay_period,admission.max_keys);
    }
    ExpertCache(const ExpertCache&)=delete;
    ExpertCache &operator=(const ExpertCache&)=delete;
    size_t resident_bytes() const {return accounting->bytes;}
    size_t size() const {return entries.size();}
    Counters counters() const {return counts;}
    size_t history_size() const {return frequency?frequency->size():0;}
    bool set_budget(size_t limit) {device_check();budget=limit;return room(0);}
    int device() const {return accounting->device;}
    size_t byte_budget() const {return budget;}
    // A failed memory sample must not evict useful hits or admit new allocations.
    // Change controls only between dispatch scopes, on the cache's host owner.
    void set_admission_enabled(bool enabled) {device_check();admission_enabled=enabled;}

    // Planning probe only. Hold PlanPins before relying on residency across
    // later misses; this does not count an access or order a GPU consumer.
    bool resident(const ExpertKey &key) const {
        device_check();key.validate();return entries.find(key)!=entries.end();
    }

    // Does not count accesses, alter frequency/LRU or upload absent keys. The
    // returned pins protect later hits from an earlier miss in the same plan.
    PlanPins protect_plan(const std::vector<ExpertKey> &plan) {
        device_check();
        if(plan.size()>12288)throw std::invalid_argument("GLM cache plan exceeds matrix limit");
        for(const auto &key:plan)key.validate();
        std::vector<std::shared_ptr<Entry>> pinned;
        std::set<ExpertKey> seen;
        for(const auto &key:plan) {
            if(!seen.insert(key).second)continue;
            auto found=entries.find(key);
            if(found!=entries.end())pinned.push_back(found->second.entry);
        }
        return PlanPins(std::move(pinned));
    }

    // Upload must enqueue all writes to the supplied stream (for example using
    // StrataExpertPipeline::transfer). It must not retain the destination or
    // enqueue writes elsewhere without ordering them onto this stream.
    template<class Upload>
    Lease get(const ExpertKey &key,std::shared_ptr<const void> source,cudaStream_t stream,Upload upload) {
        device_check();key.validate();
        auto found=entries.find(key);
        if(found!=entries.end()) {
            if(frequency)frequency->record(key);
            check(cudaStreamWaitEvent(stream,found->second.entry->ready,0));
            found->second.used=++clock;++counts.hits;
            return Lease(found->second.entry,stream);
        }
        ++counts.misses;
        if(!source)throw std::invalid_argument("GLM cache miss requires a retained source owner");
        if(frequency)frequency->record(key);
        if(!admission_enabled) {++counts.bypasses;++counts.paused_bypasses;return {};}
        if(key.bytes>budget || !room(size_t(key.bytes),&key)) {++counts.bypasses;return {};}
        std::shared_ptr<Entry> entry;
        try {entry=std::make_shared<Entry>(accounting,size_t(key.bytes),std::move(source),allocate);}
        catch(const AllocationFailure &failure) {
            if(failure.code!=cudaErrorMemoryAllocation)throw;
            // A budget sample is not a reservation. Use the existing uncached
            // transport if available memory changed before this allocation.
            // Do not clear unrelated asynchronous CUDA errors or catch failures
            // from event creation/upload. Already evicted entries stay evicted.
            if(cudaPeekAtLastError()==cudaErrorMemoryAllocation)cudaGetLastError();
            ++counts.bypasses;++counts.allocation_bypasses;return {};
        }
        try {
            upload(entry->data,size_t(key.bytes),stream);
            check(cudaEventRecord(entry->ready,stream));
        }catch(...) {cudaStreamSynchronize(stream);throw;}
        entries.emplace(key,Cached{entry,++clock});
        ++counts.admissions;
        return Lease(std::move(entry),stream);
    }
    // Explicit reload invalidation can block to drain released consumers.
    // Outstanding leases remain valid and charged to the shared byte budget;
    // a later lease release retires them. The loader must assign a fresh key
    // generation on every reload, even if paths and addresses did not change.
    size_t invalidate(const std::string &model,uint64_t generation) {
        device_check();size_t removed=0;
        if(frequency)frequency->erase_if([&](const ExpertKey &key) {
            return key.model==model && key.generation==generation;
        });
        for(auto it=entries.begin();it!=entries.end();) {
            if(it->first.model==model && it->first.generation==generation) {
                it=entries.erase(it);++removed;
            }else ++it;
        }
        counts.invalidations+=removed;return removed;
    }
};
}
