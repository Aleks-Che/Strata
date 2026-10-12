#pragma once
// Ported from backends/hy3/host_cache.hpp; private DeepSeek policy and tests.
#include "../common/expert_file.hpp"
#include <algorithm>
#include <cstring>
#include <functional>
#include <list>
#include <tuple>

namespace strata_ds4 {
// Immutable routed matrix chunks. Pageable by default; the optional CPU lane
// supplies pinned storage shared with the transfer pipeline. Keys include a live Source identity, offset and exact length
// (including MMQ padding); model reload cannot reuse another file's entries.
class HostCache {
public:
    enum class Policy { Lru, Frequency };
    struct Memory { uint64_t total, available, commit_available; };
    using Probe=std::function<Memory()>;
    struct Storage {
        std::function<uint8_t *(size_t)> allocate;
        std::function<void(uint8_t *)> release;
    };
    class Lease {
        friend class HostCache;
        std::shared_ptr<const void> owner;
        const uint8_t *pointer=nullptr;
    public:
        const void *data()const{return pointer;}
        std::shared_ptr<const void> shared_data()const{return owner?std::shared_ptr<const void>(owner,pointer):nullptr;}
        explicit operator bool()const{return bool(owner);}
    };
    struct Stats {
        uint64_t bytes=0,budget=0,entries=0,pending=0,peak_bytes=0,reused_payload_bytes=0;
        uint64_t hits=0,misses=0,hit_bytes=0,file_bytes=0,fill_bytes=0;
        uint64_t evictions=0,rejected=0,oom=0,allocations=0,reuses=0;
        uint64_t lease_hits=0,lease_misses=0,lease_bytes=0;
        uint64_t retired_bytes=0,gpu_drop_bytes=0,gpu_drop_entries=0;
        uint64_t deferred_reads=0,retain_calls=0,retain_hits=0;
        uint64_t pool_bytes=0,pool_peak=0,pool_puts=0,pool_reuses=0,pool_frees=0;
        uint64_t prefill_bypasses=0,frequency_bypasses=0,victim_candidates=0,history_entries=0;
    };
    static Memory memory() {
#ifdef _WIN32
        MEMORYSTATUSEX m{};m.dwLength=sizeof(m);
        if(!GlobalMemoryStatusEx(&m)) throw std::runtime_error("DeepSeek RAM cache memory query failed");
        return {m.ullTotalPhys,m.ullAvailPhys,m.ullAvailPageFile};
#else
        throw std::runtime_error("DeepSeek RAM cache requires a memory probe");
#endif
    }
    // Leave 7% physical RAM plus 512 MiB for driver/other-process changes.
    // The independent engine/monitor hard ceiling remains 95%.
    static uint64_t allowance(Memory m,uint64_t owned,uint64_t cap) {
        if(!m.total || m.available>m.total)throw std::runtime_error("invalid DeepSeek RAM cache memory sample");
        const uint64_t reserve=m.total/100*7+(512ULL<<20);
        const uint64_t phys=m.available>reserve?m.available-reserve:0;
        const uint64_t debt=std::max(m.available<reserve?reserve-m.available:0,
            m.commit_available<(512ULL<<20)?(512ULL<<20)-m.commit_available:0);
        const uint64_t reclaim=owned>debt?owned-debt:0;
        const uint64_t commit=m.commit_available>(512ULL<<20)?m.commit_available-(512ULL<<20):0;
        return std::min(cap,reclaim+std::min(phys,commit));
    }
private:
    using Key=std::tuple<uintptr_t,uint64_t,size_t>;
    struct Block {
        std::shared_ptr<strata_expert_file::Source> source;
        uint8_t *data=nullptr;
        size_t bytes;
        Storage storage;
        Block(std::shared_ptr<strata_expert_file::Source> s,size_t n,const Storage &allocator):source(std::move(s)),bytes(n),storage(allocator) {
            if(storage.allocate)data=storage.allocate(n);else {
#ifdef _WIN32
            data=static_cast<uint8_t *>(VirtualAlloc(nullptr,n,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
#else
            data=new(std::nothrow) uint8_t[n];
#endif
            }
            if(!data)throw std::bad_alloc();
        }
        ~Block() {
            if(storage.release){if(data)storage.release(data);return;}
#ifdef _WIN32
            if(data)VirtualFree(data,0,MEM_RELEASE);
#else
            delete[] data;
#endif
        }
    };
    using Queue=std::list<Key>;
    struct Entry {std::shared_ptr<Block> block;Queue::iterator position;bool reused=false,retired=false;};
    std::mutex mutex;
    std::map<Key,Entry> entries;
    Queue recent,retired;
    Probe probe;
    Storage storage;
    uint64_t cap,pool_limit;
    // Free payloads share the same physical budget as ready and retired entries.
    // A bounded vector avoids allocation while returning a promoted block.
    std::vector<std::shared_ptr<Block>> pool;
    static constexpr size_t pool_entries=128;
    Stats stats;
    Policy policy;
    bool prefill=false;
    struct Frequency {
        std::weak_ptr<strata_expert_file::Source> source;
        uint64_t epoch=0;
        uint8_t count=0;
    };
    // Metadata survives eviction, but never retains a file handle or payload.
    // Weak identity prevents recycled Source pointers inheriting old popularity.
    std::map<Key,Frequency> history;
    uint64_t accesses=0,decay_period;
    size_t history_limit;
    unsigned score_locked(const Key &key) const {
        const auto it=history.find(key);
        if(it==history.end() || it->second.source.expired())return 0;
        return unsigned(it->second.count)>>std::min<uint64_t>(8,accesses/decay_period-it->second.epoch);
    }
    void record_locked(const Key &key,const std::shared_ptr<strata_expert_file::Source> &source) {
        ++accesses;
        if(!history.count(key) && history.size()>=history_limit)history.clear();
        const auto value=score_locked(key);
        history[key]={source,accesses/decay_period,uint8_t(std::min(255u,value+1))};
    }
    bool frequency_room_locked(uint64_t target,const Key &key,std::shared_ptr<Block> &reuse,size_t bytes) {
        if(stats.bytes+stats.pending<=target)return true;
        // A bounded search in the LRU tail avoids a full cache scan per miss.
        // Reserve the entire victim set before changing residency: an oversized
        // or tied candidate must not evict useful entries and then get rejected.
        struct Victim {Queue::iterator position;unsigned score;};
        std::vector<Victim> victims;
        auto it=recent.end();const auto incoming=score_locked(key);
        for(size_t examined=0;it!=recent.begin() && examined<64;++examined) {
            --it;++stats.victim_candidates;
            auto found=entries.find(*it);const auto score=score_locked(*it);
            if(found->second.block.use_count()==1 && score<incoming)victims.push_back({it,score});
        }
        std::stable_sort(victims.begin(),victims.end(),[](const auto &a,const auto &b){return a.score<b.score;});
        uint64_t freed=0;size_t count=0;
        const auto needed=stats.bytes+stats.pending-target;
        while(count<victims.size() && freed<needed)freed+=entries.find(*victims[count++].position)->second.block->bytes;
        if(freed<needed)return false;
        for(size_t i=0;i<count;++i) {
            auto position=victims[i].position;auto found=entries.find(*position);
            if(!reuse && found->second.block->bytes==bytes)reuse=found->second.block;
            if(found->second.reused)stats.reused_payload_bytes-=found->second.block->bytes;
            stats.bytes-=found->second.block->bytes;++stats.evictions;
            entries.erase(found);recent.erase(position);
        }
        return true;
    }
    void trim_pool_locked(uint64_t target) {
        while(!pool.empty() && stats.bytes+stats.pending>target) {
            const auto n=pool.back()->bytes;stats.pool_bytes-=n;stats.bytes-=n;++stats.pool_frees;
            pool.pop_back();
        }
    }
    void pool_locked(std::shared_ptr<Block> block) {
        if(!pool_limit || block.use_count()!=1 || pool.size()>=pool_entries ||
           block->bytes>pool_limit-stats.pool_bytes || stats.bytes+stats.pending+block->bytes>stats.budget)return;
        const auto n=block->bytes;block->source.reset();pool.push_back(std::move(block));
        stats.bytes+=n;stats.pool_bytes+=n;++stats.pool_puts;stats.pool_peak=std::max(stats.pool_peak,stats.pool_bytes);
    }
    void trim_locked(uint64_t target,std::shared_ptr<Block> *reuse=nullptr,size_t bytes=0) {
        trim_pool_locked(target);
        auto it=recent.end();
        while(stats.bytes+stats.pending>target && it!=recent.begin()) {
            --it;auto found=entries.find(*it);
            // A producer may be copying this block into pinned staging.
            if(found->second.block.use_count()!=1)continue;
            // Reuse an already committed allocation for the incoming chunk.
            // Memory-pressure trimming never retains a spare allocation.
            if(reuse && !*reuse && found->second.block->bytes==bytes)*reuse=found->second.block;
            if(found->second.reused)stats.reused_payload_bytes-=found->second.block->bytes;
            stats.bytes-=found->second.block->bytes;++stats.evictions;
            entries.erase(found);it=recent.erase(it);
        }
    }
    void reap_locked() {
        for(auto it=retired.begin();it!=retired.end();) {
            auto found=entries.find(*it);auto &entry=found->second;
            if(entry.block.use_count()!=1){++it;continue;}
            const auto n=entry.block->bytes;
            if(entry.reused)stats.reused_payload_bytes-=n;
            stats.bytes-=n;stats.retired_bytes-=n;stats.gpu_drop_bytes+=n;++stats.gpu_drop_entries;
            auto block=std::move(entry.block);
            entries.erase(found);it=retired.erase(it);pool_locked(std::move(block));
        }
    }
    void refresh_locked() {
        reap_locked();
        stats.budget=allowance(probe(),stats.bytes+stats.pending,cap);
        trim_locked(stats.budget);
    }
public:
    explicit HostCache(uint64_t limit,Probe sample=memory,Policy choice=Policy::Frequency,
                       uint64_t decay=131072,size_t max_history=131072,Storage allocator={},uint64_t spare_limit=0)
        :probe(std::move(sample)),storage(std::move(allocator)),cap(limit),pool_limit(std::min(limit,spare_limit)),policy(choice),decay_period(std::max<uint64_t>(1,decay)),
         history_limit(std::max<size_t>(1,max_history)) {
        if(bool(storage.allocate)!=bool(storage.release))throw std::runtime_error("incomplete RAM cache allocator");
        // Never grow the cache because Windows paged out earlier cache pages.
        // Startup physical/commit headroom is a fixed upper bound; subsequent
        // probes can shrink it and restore it, but cannot increase this cap.
        cap=allowance(probe(),0,cap);pool_limit=std::min(pool_limit,cap);
        if(pool_limit)pool.reserve(pool_entries);
        refresh();
    }
    HostCache(const HostCache&)=delete;
    // The owner changes phase only after draining the route pipeline. Batched
    // MTP verification is decode, irrespective of its token count.
    void set_prefill(bool enabled) {std::lock_guard<std::mutex> lock(mutex);prefill=enabled;}
    void refresh() {std::lock_guard<std::mutex> lock(mutex);refresh_locked();}
    Stats snapshot() {
        std::lock_guard<std::mutex> lock(mutex);reap_locked();auto s=stats;s.entries=entries.size();s.history_entries=history.size();return s;
    }
    void reset_counters() {
        std::lock_guard<std::mutex> lock(mutex);
        auto s=stats;stats={};stats.bytes=s.bytes;stats.budget=s.budget;stats.pending=s.pending;stats.peak_bytes=s.bytes;
        stats.reused_payload_bytes=s.reused_payload_bytes;stats.retired_bytes=s.retired_bytes;
        stats.pool_bytes=s.pool_bytes;stats.pool_peak=s.pool_bytes;
    }
    // Drop ready RAM copies after a full matrix has been uploaded into VRAM.
    // The caller has finished the associated pipeline reads. This is a one-shot
    // promotion policy, not a permanent GPU residency registry or D2H demotion.
    // Held CPU/H2D leases remain charged and immutable until their last reader.
    void discard(const std::shared_ptr<strata_expert_file::Source> &source,uint64_t offset,size_t bytes) {
        if(!source || !bytes || !source->contains(offset,bytes))return;
        std::lock_guard<std::mutex> lock(mutex);
        const auto identity=uintptr_t(source.get());const auto end=offset+bytes;
        for(auto it=entries.lower_bound(Key{identity,offset,0});it!=entries.end() &&
            std::get<0>(it->first)==identity && std::get<1>(it->first)<end;++it) {
            auto &entry=it->second;
            if(entry.retired || entry.block->bytes>end-std::get<1>(it->first))continue;
            entry.retired=true;stats.retired_bytes+=entry.block->bytes;
            retired.splice(retired.end(),recent,entry.position);
        }
        reap_locked();
    }
    // Only return already published data; never read a file or allocate here.
    // A matrix can use a padded pipeline block, but never a partial chunk.
    // The lease prevents eviction/reuse until the CPU job has completed.
    Lease acquire(const std::shared_ptr<strata_expert_file::Source> &source,uint64_t offset,size_t bytes,bool phase_prefill=false) {
        if(!source || !bytes || !source->contains(offset,bytes))return {};
        std::lock_guard<std::mutex> lock(mutex);
        reap_locked();
        auto it=entries.lower_bound(Key{uintptr_t(source.get()),offset,bytes});
        if(it==entries.end() || std::get<0>(it->first)!=uintptr_t(source.get()) || std::get<1>(it->first)!=offset || it->second.retired) {
            ++stats.lease_misses;return {};
        }
        // The lower bound guarantees a block at least as wide as requested.
        auto &entry=it->second;Lease result;result.owner=entry.block;result.pointer=entry.block->data;
        if(!entry.reused){entry.reused=true;stats.reused_payload_bytes+=entry.block->bytes;}
        const bool is_prefill=prefill || phase_prefill;
        if(policy==Policy::Lru || !is_prefill)recent.splice(recent.begin(),recent,entry.position);
        if(policy==Policy::Frequency && !is_prefill)
            try {record_locked(it->first,source);}catch(const std::bad_alloc &){++stats.oom;}
        ++stats.lease_hits;stats.lease_bytes+=bytes;return result;
    }
private:
    // Called with the mutex held. Training happens at delivery, once only;
    // a later retain decision uses the resulting score without recording again.
    bool reserve_locked(const Key &key,size_t bytes,bool is_prefill,bool tracked,
                        std::shared_ptr<Block> &block) {
        refresh_locked();
        const auto found=entries.find(key);
        if(found!=entries.end())return false; // includes retired live leases
        bool admitted=false;
        if(policy==Policy::Frequency && is_prefill)++stats.prefill_bypasses;
        else if(policy==Policy::Frequency && (!tracked || score_locked(key)<2))++stats.frequency_bypasses;
        else if(bytes<=stats.budget) {
            // Prefer an exact-size spare, then release other free blocks before
            // considering eviction of useful cached payloads.
            for(auto it=pool.begin();it!=pool.end();++it)if((*it)->bytes==bytes) {
                block=std::move(*it);pool.erase(it);stats.pool_bytes-=bytes;stats.bytes-=bytes;++stats.pool_reuses;break;
            }
            trim_pool_locked(stats.budget-bytes);
            if(policy==Policy::Lru) {
                trim_locked(stats.budget-bytes,&block,bytes);
                admitted=stats.bytes+stats.pending<=stats.budget-bytes;
            } else {
                try {admitted=frequency_room_locked(stats.budget-bytes,key,block,bytes);}
                catch(const std::bad_alloc &) {++stats.oom;}
                if(!admitted)++stats.frequency_bypasses;
            }
        }
        if(admitted)stats.pending+=bytes;else ++stats.rejected;
        return admitted;
    }
    // A successful reservation is always released, even if custom allocation
    // fails. Payload is published only after the complete synchronous copy.
    void publish(const Key &key,const std::shared_ptr<strata_expert_file::Source> &source,
                 const void *payload,size_t bytes,const std::atomic<bool> &cancel,std::shared_ptr<Block> block) {
        bool ready=false;
        try {
            if(!cancel.load()) {
                if(block) {
                    block->source=source;
                    std::lock_guard<std::mutex> lock(mutex);++stats.reuses;
                } else {
                    block=std::make_shared<Block>(source,bytes,storage);
                    std::lock_guard<std::mutex> lock(mutex);++stats.allocations;
                }
                std::memcpy(block->data,payload,bytes);ready=true;
            }
        } catch(const std::bad_alloc &) {std::lock_guard<std::mutex> lock(mutex);++stats.oom;}
        catch(...) {std::lock_guard<std::mutex> lock(mutex);stats.pending-=bytes;throw;}
        std::lock_guard<std::mutex> lock(mutex);stats.pending-=bytes;
        if(ready && !cancel.load() && !entries.count(key)) {
            bool queued=false;
            try {
                recent.push_front(key);queued=true;
                entries.emplace(key,Entry{block,recent.begin()});
                stats.bytes+=bytes;stats.fill_bytes+=bytes;stats.peak_bytes=std::max(stats.peak_bytes,stats.bytes);
            } catch(const std::bad_alloc &) {if(queued)recent.pop_front();++stats.oom;}
        }
    }
public:
    // Loader finishes all I/O before returning; false/throw never publishes a
    // partial entry. Deferred delivery reads existing hits and trains demand,
    // but leaves admission to retain(), after the GPU allocation decision.
    template<class Loader>
    bool read(const std::shared_ptr<strata_expert_file::Source> &source,uint64_t offset,
              void *destination,size_t bytes,const std::atomic<bool> &cancel,Loader load,
              bool phase_prefill=false,bool defer_admission=false) {
        if(!source || !source->contains(offset,bytes) || (bytes && !destination))
            throw std::runtime_error("invalid DeepSeek RAM cache range");
        if(cancel.load())return false;
        if(!bytes)return true;
        const Key key{uintptr_t(source.get()),offset,bytes};
        std::shared_ptr<Block> hit,block;
        bool admitted=false;
        {
            std::lock_guard<std::mutex> lock(mutex);reap_locked();
            const bool is_prefill=prefill || phase_prefill;
            bool tracked=true;
            if(policy==Policy::Frequency && !is_prefill) {
                try {record_locked(key,source);}catch(const std::bad_alloc &) {tracked=false;++stats.oom;}
            }
            auto found=entries.find(key);
            if(found!=entries.end() && !found->second.retired) {
                hit=found->second.block;
                if(!found->second.reused) {found->second.reused=true;stats.reused_payload_bytes+=bytes;}
                if(policy==Policy::Lru || !is_prefill)recent.splice(recent.begin(),recent,found->second.position);
                ++stats.hits;stats.hit_bytes+=bytes;
            } else {
                ++stats.misses;
                if(defer_admission)++stats.deferred_reads;
                else admitted=reserve_locked(key,bytes,is_prefill,tracked,block);
            }
        }
        if(hit) {std::memcpy(destination,hit->data,bytes);return !cancel.load();}
        bool ok=false;
        try {ok=load(destination,bytes);} catch(...) {
            if(admitted) {std::lock_guard<std::mutex> lock(mutex);stats.pending-=bytes;}
            throw;
        }
        if(ok) {std::lock_guard<std::mutex> lock(mutex);stats.file_bytes+=bytes;}
        if(admitted) {
            if(ok)publish(key,source,destination,bytes,cancel,std::move(block));
            else {std::lock_guard<std::mutex> lock(mutex);stats.pending-=bytes;}
        }
        return ok && !cancel.load();
    }
    // Immutable, already delivered bytes owned by the caller until return.
    // No file I/O, no extra training and no change to an existing ready block.
    void retain(const std::shared_ptr<strata_expert_file::Source> &source,uint64_t offset,
                const void *payload,size_t bytes,const std::atomic<bool> &cancel,bool phase_prefill=false) {
        if(!source || !source->contains(offset,bytes) || (bytes && !payload))
            throw std::runtime_error("invalid DeepSeek RAM retain range");
        if(!bytes || cancel.load())return;
        const Key key{uintptr_t(source.get()),offset,bytes};std::shared_ptr<Block> block;
        {
            std::lock_guard<std::mutex> lock(mutex);++stats.retain_calls;reap_locked();
            if(prefill || phase_prefill){++stats.prefill_bypasses;return;}
            if(entries.count(key)){++stats.retain_hits;return;}
            if(!reserve_locked(key,bytes,false,true,block))return;
        }
        publish(key,source,payload,bytes,cancel,std::move(block));
    }

    bool read(const std::shared_ptr<strata_expert_file::Source> &source,uint64_t offset,
              void *destination,size_t bytes,const std::atomic<bool> &cancel,strata_expert_file::Request &request) {
        return read(source,offset,destination,bytes,cancel,[&](void *p,size_t n){return request.read_at(*source,offset,p,n,cancel);});
    }
};
}
