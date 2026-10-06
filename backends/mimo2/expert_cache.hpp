#pragma once
// Single owner, synchronous cache. No pointer may outlive its completed copy.
// Fill callbacks must fence on success AND failure before returning/throwing.
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <list>
#include <iterator>
#include <map>
#include <stdexcept>
#include <tuple>
#include <memory>
#include <vector>
namespace mimo2 {
struct MatrixKey {
    uint64_t generation;uint32_t tensor,expert;
    bool operator<(const MatrixKey &b) const {return std::tie(generation,tensor,expert)<std::tie(b.generation,b.tensor,b.expert);}
};
class ExpertCache {
public:
    using Allocate=std::function<void *(size_t)>;
    using Free=std::function<void(void *)>;
    struct Counters {uint64_t hits=0,misses=0,evictions=0,allocations=0,reuses=0,bypasses=0,oom=0;};
private:
    struct Entry {void *data;size_t bytes,charge;std::list<MatrixKey>::iterator order;};
    std::map<MatrixKey,Entry> entries;std::list<MatrixKey> order;
    std::map<MatrixKey,size_t> pins;
    size_t cap,limit=0,resident=0,payload=0,granularity;Allocate allocate;Free release;Counters counters;
    void erase(std::map<MatrixKey,Entry>::iterator it) {
        release(it->second.data);resident-=it->second.charge;payload-=it->second.bytes;
        order.erase(it->second.order);entries.erase(it);++counters.evictions;
    }
public:
    // Protect only entries already resident when the route is planned. New
    // fills are fenced synchronously and have no outstanding cache consumers.
    class PlanPins {
        ExpertCache &cache;std::vector<MatrixKey> keys;
    public:
        PlanPins(ExpertCache &cache,const std::vector<MatrixKey> &requested):cache(cache) {
            try {for(const auto &key:requested)if(cache.entries.count(key)) {
                keys.push_back(key);++cache.pins[key];
            }} catch(...) {release();throw;}
        }
        PlanPins(const PlanPins &)=delete;
        void release() {for(const auto &key:keys) {auto it=cache.pins.find(key);if(it!=cache.pins.end() && !--it->second)cache.pins.erase(it);}keys.clear();}
        ~PlanPins() {release();}
    };
    std::unique_ptr<PlanPins> protect(const std::vector<MatrixKey> &keys) {return std::make_unique<PlanPins>(*this,keys);}
    bool contains(const MatrixKey &key,size_t bytes) const {
        auto it=entries.find(key);if(it==entries.end())return false;
        if(it->second.bytes!=bytes)throw std::runtime_error("cache identity changed matrix size");return true;
    }
    ExpertCache(size_t cap,Allocate allocate,Free release,size_t granularity=65536):cap(cap),granularity(granularity),allocate(allocate),release(release) {
        if(!granularity)throw std::runtime_error("zero cache allocation granularity");
    }
    ~ExpertCache() {for(auto &e:entries)release(e.second.data);}
    ExpertCache(const ExpertCache &)=delete;ExpertCache &operator=(const ExpertCache &)=delete;
    static size_t charge(size_t bytes,size_t page=65536) {
        if(!page)throw std::runtime_error("zero allocation granularity");
        if(!bytes || bytes>SIZE_MAX-(page-1))throw std::runtime_error("invalid cache matrix size");
        return ((bytes+page-1)/page)*page; // cudaMalloc is explicitly asked for this rounded size.
    }
    size_t bytes() const {return resident;}
    size_t payload_bytes() const {return payload;}
    size_t budget() const {return limit;}
    size_t requested() const {return cap;}
    Counters stats() const {return counters;}
    void reset_stats() {counters={};}
    void trim(size_t bytes) {
        for(auto it=order.begin();resident>bytes && it!=order.end();) {
            auto key=*it++;if(!pins.count(key))erase(entries.find(key));
        }
    }
    void constrain(size_t free,size_t total) {
        if(!total || free>total)throw std::runtime_error("invalid cache memory sample");
        // 5% remains outside the application; 256 MiB absorbs driver/sample lag.
        const size_t reserve=(total+19)/20+(256ull<<20);
        const size_t available=free>=reserve?resident+free-reserve:resident-std::min(resident,reserve-free);
        limit=std::min(cap,available);trim(limit);
    }
    void *get(const MatrixKey &key,size_t bytes,bool touch=true) {
        auto it=entries.find(key);
        if(it==entries.end()) {++counters.misses;return nullptr;}
        if(it->second.bytes!=bytes)throw std::runtime_error("cache identity changed matrix size");
        ++counters.hits;if(touch)order.splice(order.end(),order,it->second.order);
        return it->second.data;
    }
    // Publish only after a successful, fenced fill. Cancel/error cannot cache partial data.
    void store(const MatrixKey &key,size_t bytes,const std::function<void(void *)> &fill) {
        if(entries.count(key))throw std::runtime_error("duplicate cache store");
        const size_t charged=charge(bytes,granularity);
        if(charged>limit) {++counters.bypasses;return;}
        void *data=nullptr;
        if(resident>limit-charged) {
            // Reuse a compatible allocation among old entries; no cudaMalloc in a warm steady state.
            size_t scanned=0;
            for(auto it=order.begin();it!=order.end() && scanned++<64;++it) {
                auto victim=entries.find(*it);
                if(victim->second.charge==charged && !pins.count(victim->first)) {
                    data=victim->second.data;resident-=charged;payload-=victim->second.bytes;entries.erase(victim);order.erase(it);
                    ++counters.evictions;++counters.reuses;break;
                }
            }
        }
        trim(limit-charged);
        if(resident>limit-charged) {if(data)release(data);++counters.bypasses;return;}
        if(!data) {data=allocate(charged);if(!data) {++counters.oom;return;}++counters.allocations;}
        resident+=charged;
        try {
            fill(data);
            order.push_back(key);
            try {entries.emplace(key,Entry{data,bytes,charged,std::prev(order.end())});}
            catch(...) {order.pop_back();throw;}
            payload+=bytes;
        } catch(...) {release(data);resident-=charged;throw;}
    }
};
} // namespace mimo2
