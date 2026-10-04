#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace strata_glm {
enum class Branch { main, mtp };
enum class Projection { gate, up, down };

// Native equivalent of ExpertMatrix.cache_key in tools/glm5next_expert_plan.py.
// Identity refers to the complete model, not its architecture or mmap address.
struct ExpertKey {
    std::string model;
    uint64_t generation;
    Branch branch;
    int layer,expert;
    Projection projection;
    std::string quant;
    uint64_t columns,rows;
    std::string shard;
    uint64_t offset,bytes;
    auto fields() const {
        return std::tie(model,generation,branch,layer,expert,projection,quant,
                        columns,rows,shard,offset,bytes);
    }
    bool operator<(const ExpertKey &other) const {return fields()<other.fields();}
    bool operator==(const ExpertKey &other) const {return fields()==other.fields();}
    void validate() const {
        if(model.empty() || shard.empty() || quant.empty() || layer<0 || expert<0 ||
           (branch!=Branch::main && branch!=Branch::mtp) ||
           (projection!=Projection::gate && projection!=Projection::up && projection!=Projection::down) ||
           !columns || !rows || !bytes || bytes>std::numeric_limits<size_t>::max() ||
           offset>std::numeric_limits<uint64_t>::max()-bytes)
            throw std::invalid_argument("invalid GLM expert cache key");
    }
};

// Single host owner, one CUDA device. A lease protects a plan's matrices from
// eviction. Queue all consumers on its stream before release/destruction.
// A miss can bypass when all eviction candidates are leased or still in flight.
class ExpertCache {
    static void check(cudaError_t e) {
        if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    }
    struct Accounting {size_t bytes=0;int device;explicit Accounting(int d):device(d) {}};
    struct Entry {
        std::shared_ptr<Accounting> accounting;
        std::shared_ptr<const void> source;
        void *data=nullptr;
        size_t bytes;
        cudaEvent_t ready=nullptr;
        std::vector<cudaEvent_t> consumers;
        Entry(std::shared_ptr<Accounting> a,size_t n,std::shared_ptr<const void> owner)
            :accounting(std::move(a)),source(std::move(owner)),bytes(n) {
            check(cudaMalloc(&data,bytes));
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
    struct Counters {uint64_t hits=0,misses=0,bypasses=0,evictions=0,invalidations=0;};
private:
    struct Cached {std::shared_ptr<Entry> entry;uint64_t used;};
    std::map<ExpertKey,Cached> entries;
    std::shared_ptr<Accounting> accounting;
    size_t budget;
    uint64_t clock=0;
    Counters counts;
    void device_check() const {
        int device=-1;check(cudaGetDevice(&device));
        if(device!=accounting->device)throw std::runtime_error("GLM cache used on another CUDA device");
    }
    bool room(size_t wanted) {
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
    explicit ExpertCache(size_t limit):budget(limit) {
        int device=-1;check(cudaGetDevice(&device));accounting=std::make_shared<Accounting>(device);
    }
    ExpertCache(const ExpertCache&)=delete;
    ExpertCache &operator=(const ExpertCache&)=delete;
    size_t resident_bytes() const {return accounting->bytes;}
    size_t size() const {return entries.size();}
    Counters counters() const {return counts;}
    bool set_budget(size_t limit) {device_check();budget=limit;return room(0);}

    // Upload must enqueue all writes to the supplied stream (for example using
    // StrataExpertPipeline::transfer). It must not retain the destination or
    // enqueue writes elsewhere without ordering them onto this stream.
    template<class Upload>
    Lease get(const ExpertKey &key,std::shared_ptr<const void> source,cudaStream_t stream,Upload upload) {
        device_check();key.validate();
        auto found=entries.find(key);
        if(found!=entries.end()) {
            check(cudaStreamWaitEvent(stream,found->second.entry->ready,0));
            found->second.used=++clock;++counts.hits;
            return Lease(found->second.entry,stream);
        }
        ++counts.misses;
        if(!source)throw std::invalid_argument("GLM cache miss requires a retained source owner");
        if(key.bytes>budget || !room(size_t(key.bytes))) {++counts.bypasses;return {};}
        auto entry=std::make_shared<Entry>(accounting,size_t(key.bytes),std::move(source));
        try {
            upload(entry->data,size_t(key.bytes),stream);
            check(cudaEventRecord(entry->ready,stream));
        }catch(...) {cudaStreamSynchronize(stream);throw;}
        entries.emplace(key,Cached{entry,++clock});
        return Lease(std::move(entry),stream);
    }
    // Explicit reload invalidation can block to drain released consumers.
    // Outstanding leases remain valid and charged to the shared byte budget;
    // a later lease release retires them. The loader must assign a fresh key
    // generation on every reload, even if paths and addresses did not change.
    size_t invalidate(const std::string &model,uint64_t generation) {
        device_check();size_t removed=0;
        for(auto it=entries.begin();it!=entries.end();) {
            if(it->first.model==model && it->first.generation==generation) {
                it=entries.erase(it);++removed;
            }else ++it;
        }
        counts.invalidations+=removed;return removed;
    }
};
}
