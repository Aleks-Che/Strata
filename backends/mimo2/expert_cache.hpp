#pragma once
// Single owner. Synchronous fills fence on success AND failure. FillBatch
// reserves invisible entries, then submits live reservations and fences once.
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <exception>
#include <list>
#include <iterator>
#include <unordered_map>
#include <stdexcept>
#include <tuple>
#include <memory>
#include <vector>
#include "../common/expert_frequency.hpp"
namespace mimo2 {
struct MatrixKey {
    uint64_t generation;uint32_t tensor,expert;
    bool operator<(const MatrixKey &b) const {return std::tie(generation,tensor,expert)<std::tie(b.generation,b.tensor,b.expert);}
    bool operator==(const MatrixKey &b) const {return std::tie(generation,tensor,expert)==std::tie(b.generation,b.tensor,b.expert);}
};
struct MatrixHash {
    size_t operator()(const MatrixKey &key) const {
        // MSVC uses power-of-two bucket counts. Keeping the tensor only above
        // bit 15 puts every layer's same expert in one history bucket at the
        // usual table size. Mix all identity fields into the low bits too.
        uint64_t value=key.generation*0x9e3779b97f4a7c15ULL;
        value^=(uint64_t(key.tensor)<<32)|key.expert;
        value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
        value=(value^(value>>27))*0x94d049bb133111ebULL;
        return size_t(value^(value>>31));
    }
};
class ExpertCache {
public:
    using Allocate=std::function<void *(size_t)>;
    using Free=std::function<void(void *)>;
    struct Counters {uint64_t hits=0,misses=0,evictions=0,allocations=0,reuses=0,bypasses=0,oom=0;
        uint64_t frequency_updates=0,frequency_rejected=0,frequency_candidates=0;};
private:
    struct Entry;
    using Order=std::list<Entry *>;
    struct Entry {MatrixKey key;void *data;size_t bytes,charge;Order::iterator order;uint64_t pending;size_t pins=0;};
    // Rehash invalidates iterators, but not pointers to unordered_map elements.
    // LRU and route pins retain those pointers; neither stores map iterators.
    std::unordered_map<MatrixKey,Entry,MatrixHash> entries;Order order;
    size_t cap,limit=0,resident=0,payload=0,granularity;Allocate allocate;Free release;Counters counters;
    std::function<size_t()> physical_bytes;
    std::function<size_t(size_t)> physical_growth;
    uint64_t decay;
    std::unique_ptr<StrataExpertFrequencyHistory<MatrixKey,MatrixHash>> history;
    bool batch_active=false;
    uint64_t next_ticket=0;
    size_t pending_entries=0;
    // Oldest 64 eligible entries, lowest frequency first; LRU breaks ties.
    // Prefer a compatible allocation so slab holes in other classes need not
    // force release of a whole block. Pins always override admission policy.
    Entry *victim(size_t matching_charge=0) {
        Entry *best=nullptr;unsigned score=0;size_t scanned=0;
        for(auto *entry:order) {
            if(entry->pins)continue;
            if(scanned++==64)break;
            if(matching_charge && entry->charge!=matching_charge)continue;
            ++counters.frequency_candidates;
            const auto value=history->score(entry->key);
            if(!best || value<score) {best=entry;score=value;}
            if(!score)break;
        }
        return best;
    }
    bool colder(const MatrixKey &key,const MatrixKey &other) {
        if(history->score(key)>=history->score(other))return false;
        ++counters.frequency_rejected;return true;
    }
    bool fits(size_t charge) const {
        const size_t used=bytes(),extra=physical_growth?physical_growth(charge):charge;
        return used<=limit && extra<=limit-used;
    }
    void *detach(Entry *entry) {
        const auto key=entry->key;void *data=entry->data;
        resident-=entry->charge;payload-=entry->bytes;
        if(entry->pending)--pending_entries;
        order.erase(entry->order);entries.erase(key);++counters.evictions;
        return data;
    }
    void erase(Entry *entry) {release(entry->data);detach(entry);}
public:
    // Protect only completed entries already resident when the route is planned.
    class PlanPins {
        std::vector<Entry *> entries;
    public:
        PlanPins(ExpertCache &cache,const std::vector<MatrixKey> &requested) {
            entries.reserve(requested.size());
            try {for(const auto &key:requested) {
                auto it=cache.entries.find(key);if(it==cache.entries.end() || it->second.pending)continue;
                entries.push_back(&it->second);++it->second.pins;
            }} catch(...) {release();throw;}
        }
        PlanPins(const PlanPins &)=delete;
        void release() {for(auto *entry:entries)--entry->pins;entries.clear();}
        ~PlanPins() {release();}
    };
    std::unique_ptr<PlanPins> protect(const std::vector<MatrixKey> &keys) {return std::make_unique<PlanPins>(*this,keys);}
    bool contains(const MatrixKey &key,size_t bytes) const {
        auto it=entries.find(key);if(it==entries.end())return false;
        if(it->second.bytes!=bytes)throw std::runtime_error("cache identity changed matrix size");return !it->second.pending;
    }
    ExpertCache(size_t cap,Allocate allocate,Free release,size_t granularity=65536,
                std::function<size_t()> physical_bytes={},std::function<size_t(size_t)> physical_growth={},uint64_t decay_period=0)
        :cap(cap),granularity(granularity),allocate(allocate),release(release),
         physical_bytes(std::move(physical_bytes)),physical_growth(std::move(physical_growth)),decay(decay_period) {
        if(!granularity)throw std::runtime_error("zero cache allocation granularity");
        if(bool(this->physical_bytes)!=bool(this->physical_growth))throw std::runtime_error("incomplete physical allocator contract");
        if(decay)history=std::make_unique<StrataExpertFrequencyHistory<MatrixKey,MatrixHash>>(decay,65536);
    }
    ~ExpertCache() {for(auto &e:entries)release(e.second.data);}
    ExpertCache(const ExpertCache &)=delete;ExpertCache &operator=(const ExpertCache &)=delete;
    static size_t charge(size_t bytes,size_t page=65536) {
        if(!page)throw std::runtime_error("zero allocation granularity");
        if(!bytes || bytes>SIZE_MAX-(page-1))throw std::runtime_error("invalid cache matrix size");
        return ((bytes+page-1)/page)*page; // cudaMalloc is explicitly asked for this rounded size.
    }
    size_t bytes() const {return physical_bytes?physical_bytes():resident;}
    size_t slot_bytes() const {return resident;}
    size_t payload_bytes() const {return payload;}
    size_t budget() const {return limit;}
    size_t requested() const {return cap;}
    size_t pending() const {return pending_entries;}
    Counters stats() const {return counters;}
    uint64_t decay_period() const {return decay;}
    size_t history_keys() const {return history?history->size():0;}
    // One actual selected matrix, including misses; callers exclude guard tails
    // and prefill when admission is disabled. Request stats reset keeps history.
    void record(const MatrixKey &key) {if(history) {history->record(key);++counters.frequency_updates;}}
    void reset_stats() {counters={};}
    void trim(size_t bytes) {
        for(auto it=order.begin();this->bytes()>bytes && it!=order.end();) {
            auto *entry=*it++;if(!entry->pins)erase(entry);
        }
    }
    void constrain(size_t free,size_t total) {
        if(!total || free>total)throw std::runtime_error("invalid cache memory sample");
        // 5% remains outside the application; 256 MiB absorbs driver/sample lag.
        const size_t reserve=(total+19)/20+(256ull<<20);
        const size_t used=bytes();
        const size_t available=free>=reserve?used+free-reserve:used-std::min(used,reserve-free);
        limit=std::min(cap,available);trim(limit);
    }
    void *get(const MatrixKey &key,size_t bytes,bool touch=true) {
        auto it=entries.find(key);
        if(it==entries.end()) {++counters.misses;return nullptr;}
        if(it->second.bytes!=bytes)throw std::runtime_error("cache identity changed matrix size");
        if(it->second.pending) {++counters.misses;return nullptr;}
        ++counters.hits;if(touch)order.splice(order.end(),order,it->second.order);
        return it->second.data;
    }
    // Replay LRU order after a tensor's delivery fence without a second lookup
    // being counted as a cache hit. The caller retains route pins throughout.
    void touch(const MatrixKey &key) {
        auto it=entries.find(key);
        if(it==entries.end() || it->second.pending)throw std::runtime_error("touch of absent or pending cache entry");
        order.splice(order.end(),order,it->second.order);
    }
    // Publish only after a successful, fenced fill. Cancel/error cannot cache partial data.
    void store(const MatrixKey &key,size_t bytes,const std::function<void(void *)> &fill) {
        if(batch_active)throw std::runtime_error("synchronous store during cache fill batch");
        store_impl(key,bytes,fill,0);
    }
    // No GPU work is submitted during admission. Pending reservations participate
    // in the original LRU/frequency decisions and may be evicted safely. Tickets
    // distinguish repeated keys and reused addresses. Only survivors are copied.
    // Submit callbacks only enqueue copies; fence MUST drain even when throwing.
    // The batch and its sources must outlive finish(); no cache mutation may run
    // inside submit/fence. Destructor discards unsubmitted reservations on cancel.
    class FillBatch {
        struct Work {MatrixKey key;uint64_t ticket;std::function<void(void *)> submit;};
        ExpertCache &cache;std::function<void()> fence;std::vector<Work> work;bool closed=false;
        bool matches(const Work &w) const {
            auto it=cache.entries.find(w.key);return it!=cache.entries.end() && it->second.pending==w.ticket;
        }
        void discard() {
            if(closed)return;
            for(const auto &w:work)if(matches(w))cache.erase(&cache.entries.find(w.key)->second);
            cache.batch_active=false;closed=true;
        }
    public:
        FillBatch(ExpertCache &cache,std::function<void()> fence,size_t capacity=0):cache(cache),fence(std::move(fence)) {
            if(cache.batch_active || !this->fence)throw std::runtime_error("invalid or nested cache fill batch");
            work.reserve(capacity);
            cache.batch_active=true;
        }
        FillBatch(const FillBatch &)=delete;
        ~FillBatch() {discard();}
        void store(const MatrixKey &key,size_t bytes,const std::function<void(void *)> &submit) {
            if(closed || !submit || cache.next_ticket==UINT64_MAX)throw std::runtime_error("invalid cache fill reservation");
            const auto ticket=++cache.next_ticket;
            cache.store_impl(key,bytes,[&](void *) {work.push_back({key,ticket,submit});},ticket);
        }
        void finish() {
            if(closed)throw std::runtime_error("cache fill batch already closed");
            bool submitted=false,fenced=false;
            try {
                for(const auto &w:work)if(matches(w)) {
                    submitted=true;w.submit(cache.entries.find(w.key)->second.data);
                }
                if(submitted) {fenced=true;fence();}
                for(const auto &w:work)if(matches(w)) {
                    cache.entries.find(w.key)->second.pending=0;--cache.pending_entries;
                }
                cache.batch_active=false;closed=true;
            } catch(...) {
                const auto error=std::current_exception();
                if(submitted && !fenced)try {fence();}catch(...) {discard();throw;}
                discard();std::rethrow_exception(error);
            }
        }
    };
private:
    void store_impl(const MatrixKey &key,size_t bytes,const std::function<void(void *)> &fill,uint64_t ticket) {
        if(entries.count(key))throw std::runtime_error("duplicate cache store");
        const size_t charged=charge(bytes,granularity);
        if(charged>limit) {++counters.bypasses;return;}
        void *data=nullptr;
        if(!fits(charged) && history) {
            auto selected=victim(charged);
            if(selected) {
                if(colder(key,selected->key))return;
                data=detach(selected);++counters.reuses;
            }
        } else if(!fits(charged)) {
            // Reuse a compatible allocation among old entries; no cudaMalloc in a warm steady state.
            size_t scanned=0;
            for(auto it=order.begin();it!=order.end() && scanned++<64;++it) {
                auto *entry=*it;
                if(entry->charge==charged && !entry->pins) {
                    data=detach(entry);++counters.reuses;break;
                }
            }
        }
        // A reused slot stays physically reserved. Releasing other entries may
        // create a compatible hole without freeing their whole block.
        const size_t needed=data && physical_bytes?0:charged;
        if(history) {
            while(!fits(needed)) {
                auto selected=victim();if(!selected)break;
                if(!data && colder(key,selected->key))return;
                erase(selected);
            }
        } else for(auto it=order.begin();!fits(needed) && it!=order.end();) {
            auto *entry=*it++;if(!entry->pins)erase(entry);
        }
        if(!fits(needed)) {if(data)release(data);++counters.bypasses;return;}
        if(!data) {data=allocate(charged);if(!data) {++counters.oom;return;}++counters.allocations;}
        resident+=charged;
        try {
            fill(data);
            auto inserted=entries.emplace(key,Entry{key,data,bytes,charged,order.end(),ticket});
            if(!inserted.second)throw std::runtime_error("duplicate cache store during fill");
            auto &entry=inserted.first->second;
            try {order.push_back(&entry);}
            catch(...) {entries.erase(inserted.first);throw;}
            entry.order=std::prev(order.end());
            payload+=bytes;
            if(ticket)++pending_entries;
        } catch(...) {release(data);resident-=charged;throw;}
    }
};
} // namespace mimo2
