#pragma once
#include "../common/expert_file.hpp"
#include <algorithm>
#include <cstring>
#include <functional>
#include <list>
#include <tuple>

namespace hy3 {
// Pageable copies of immutable routed matrix chunks. Only the small pipeline
// ring is pinned. Keys include a live Source identity, offset and exact length
// (including MMQ padding); model reload cannot reuse another file's entries.
class HostCache {
public:
    enum class Policy { Lru, Frequency };
    struct Memory { uint64_t total, available, commit_available; };
    using Probe=std::function<Memory()>;
    struct Stats {
        uint64_t bytes=0,budget=0,entries=0,pending=0,peak_bytes=0,reused_payload_bytes=0;
        uint64_t hits=0,misses=0,hit_bytes=0,file_bytes=0,fill_bytes=0;
        uint64_t evictions=0,rejected=0,oom=0,allocations=0,reuses=0;
        uint64_t prefill_bypasses=0,frequency_bypasses=0,victim_candidates=0,history_entries=0;
    };
    static Memory memory() {
#ifdef _WIN32
        MEMORYSTATUSEX m{};m.dwLength=sizeof(m);
        if(!GlobalMemoryStatusEx(&m)) throw std::runtime_error("Hy3 RAM cache memory query failed");
        return {m.ullTotalPhys,m.ullAvailPhys,m.ullAvailPageFile};
#else
        throw std::runtime_error("Hy3 RAM cache requires a memory probe");
#endif
    }
    // Leave 7% physical RAM plus 512 MiB for driver/other-process changes.
    // The independent engine/monitor hard ceiling remains 95%.
    static uint64_t allowance(Memory m,uint64_t owned,uint64_t cap) {
        if(!m.total || m.available>m.total)throw std::runtime_error("invalid Hy3 RAM cache memory sample");
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
        Block(std::shared_ptr<strata_expert_file::Source> s,size_t n):source(std::move(s)),bytes(n) {
#ifdef _WIN32
            data=static_cast<uint8_t *>(VirtualAlloc(nullptr,n,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
#else
            data=new(std::nothrow) uint8_t[n];
#endif
            if(!data)throw std::bad_alloc();
        }
        ~Block() {
#ifdef _WIN32
            if(data)VirtualFree(data,0,MEM_RELEASE);
#else
            delete[] data;
#endif
        }
    };
    using Queue=std::list<Key>;
    struct Entry {std::shared_ptr<Block> block;Queue::iterator position;bool reused=false;};
    std::mutex mutex;
    std::map<Key,Entry> entries;
    Queue recent;
    Probe probe;
    uint64_t cap;
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
    void trim_locked(uint64_t target,std::shared_ptr<Block> *reuse=nullptr,size_t bytes=0) {
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
    void refresh_locked() {
        stats.budget=allowance(probe(),stats.bytes+stats.pending,cap);
        trim_locked(stats.budget);
    }
public:
    explicit HostCache(uint64_t limit,Probe sample=memory,Policy choice=Policy::Frequency,
                       uint64_t decay=131072,size_t max_history=131072)
        :probe(std::move(sample)),cap(limit),policy(choice),decay_period(std::max<uint64_t>(1,decay)),
         history_limit(std::max<size_t>(1,max_history)) {
        // Never grow the cache because Windows paged out earlier cache pages.
        // Startup physical/commit headroom is a fixed upper bound; subsequent
        // probes can shrink it and restore it, but cannot increase this cap.
        cap=allowance(probe(),0,cap);refresh();
    }
    HostCache(const HostCache&)=delete;
    // The owner changes phase only after draining the route pipeline. Batched
    // MTP verification is decode, irrespective of its token count.
    void set_prefill(bool enabled) {std::lock_guard<std::mutex> lock(mutex);prefill=enabled;}
    void refresh() {std::lock_guard<std::mutex> lock(mutex);refresh_locked();}
    Stats snapshot() {
        std::lock_guard<std::mutex> lock(mutex);auto s=stats;s.entries=entries.size();s.history_entries=history.size();return s;
    }
    void reset_counters() {
        std::lock_guard<std::mutex> lock(mutex);
        auto s=stats;stats={};stats.bytes=s.bytes;stats.budget=s.budget;stats.pending=s.pending;stats.peak_bytes=s.bytes;
        stats.reused_payload_bytes=s.reused_payload_bytes;
    }
    // Loader finishes all I/O before returning; false/throw never publishes a
    // partial cache entry. This overload also permits deterministic CPU tests.
    template<class Loader>
    bool read(const std::shared_ptr<strata_expert_file::Source> &source,uint64_t offset,
              void *destination,size_t bytes,const std::atomic<bool> &cancel,Loader load) {
        if(!source || !source->contains(offset,bytes) || (bytes && !destination))
            throw std::runtime_error("invalid Hy3 RAM cache range");
        if(cancel.load())return false;
        if(!bytes)return true;
        const Key key{uintptr_t(source.get()),offset,bytes};
        std::shared_ptr<Block> hit,block;
        bool admitted=false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            bool tracked=true;
            if(policy==Policy::Frequency && !prefill) {
                try {record_locked(key,source);}catch(const std::bad_alloc &) {tracked=false;++stats.oom;}
            }
            auto found=entries.find(key);
            if(found!=entries.end()) {
                hit=found->second.block;
                if(!found->second.reused) {found->second.reused=true;stats.reused_payload_bytes+=bytes;}
                if(policy==Policy::Lru || !prefill)recent.splice(recent.begin(),recent,found->second.position);
                ++stats.hits;stats.hit_bytes+=bytes;
            } else {
                ++stats.misses;refresh_locked();
                if(policy==Policy::Frequency && prefill)++stats.prefill_bypasses;
                else if(policy==Policy::Frequency && (!tracked || score_locked(key)<2))++stats.frequency_bypasses;
                else if(bytes<=stats.budget) {
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
            }
        }
        if(hit) {std::memcpy(destination,hit->data,bytes);return !cancel.load();}
        // Read directly into pinned staging: admission failure has no impact
        // on the delivery path and never retries an I/O error as a cache miss.
        bool ok=false;
        try {ok=load(destination,bytes);} catch(...) {
            if(admitted) {std::lock_guard<std::mutex> lock(mutex);stats.pending-=bytes;}
            throw;
        }
        if(ok) {std::lock_guard<std::mutex> lock(mutex);stats.file_bytes+=bytes;}
        bool ready=false;
        try {
            if(admitted && ok && !cancel.load()) {
                if(block) {
                    block->source=source;
                    std::lock_guard<std::mutex> lock(mutex);++stats.reuses;
                } else {
                    block=std::make_shared<Block>(source,bytes);
                    std::lock_guard<std::mutex> lock(mutex);++stats.allocations;
                }
                std::memcpy(block->data,destination,bytes);
                ready=true;
            }
        } catch(const std::bad_alloc &) {std::lock_guard<std::mutex> lock(mutex);++stats.oom;}
        if(admitted) {
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
        return ok && !cancel.load();
    }
    bool read(const std::shared_ptr<strata_expert_file::Source> &source,uint64_t offset,
              void *destination,size_t bytes,const std::atomic<bool> &cancel,strata_expert_file::Request &request) {
        return read(source,offset,destination,bytes,cancel,[&](void *p,size_t n){return request.read_at(*source,offset,p,n,cancel);});
    }
};
}
