#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>

// Admission history is independent of residency: a streamed matrix still gets
// another chance after repeated use. Lazy decay lets a new workload replace old
// hot entries. This is metadata only; it never changes model routing or weights.
// Key/Hash permit full model identities without changing the pointer-key API.
template<class Key, class Hash=std::hash<Key>>
class StrataExpertFrequencyHistory {
    struct Count { uint64_t epoch=0; uint8_t value=0; };
    std::unordered_map<Key,Count,Hash> counts;
    uint64_t clock=0,period;
    size_t max_keys;
public:
    explicit StrataExpertFrequencyHistory(uint64_t decay_period,size_t limit=131072)
        :period(std::max<uint64_t>(1,decay_period)),max_keys(std::max<size_t>(1,limit)) {}
    unsigned score(const Key &key) const {
        auto it=counts.find(key);
        if(it==counts.end())return 0;
        return unsigned(it->second.value)>>std::min<uint64_t>(8,clock/period-it->second.epoch);
    }
    void record(const Key &key) {
        ++clock;
        auto it=counts.find(key);
        if(it==counts.end() && counts.size()>=max_keys)counts.clear();
        unsigned value=score(key);
        counts[key]={clock/period,uint8_t(std::min(255u,value+1))};
    }
    template<class Predicate>
    void erase_if(Predicate predicate) {
        for(auto it=counts.begin();it!=counts.end();) {
            if(predicate(it->first))it=counts.erase(it);
            else ++it;
        }
    }
    size_t size() const {return counts.size();}
};

using StrataExpertFrequency=StrataExpertFrequencyHistory<const void *>;
