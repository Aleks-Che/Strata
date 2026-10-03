#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

// Admission history is independent of residency: a streamed matrix still gets
// another chance after repeated use. Lazy decay lets a new workload replace old
// hot entries. This is metadata only; it never changes model routing or weights.
class StrataExpertFrequency {
    struct Count { uint64_t epoch=0; uint8_t value=0; };
    std::unordered_map<const void *,Count> counts;
    uint64_t clock=0,period;
    size_t max_keys;
public:
    explicit StrataExpertFrequency(uint64_t decay_period,size_t limit=131072)
        :period(std::max<uint64_t>(1,decay_period)),max_keys(std::max<size_t>(1,limit)) {}
    unsigned score(const void *key) const {
        auto it=counts.find(key);
        if(it==counts.end())return 0;
        return unsigned(it->second.value)>>std::min<uint64_t>(8,clock/period-it->second.epoch);
    }
    void record(const void *key) {
        ++clock;
        auto it=counts.find(key);
        if(it==counts.end() && counts.size()>=max_keys)counts.clear();
        unsigned value=score(key);
        counts[key]={clock/period,uint8_t(std::min(255u,value+1))};
    }
    size_t size() const {return counts.size();}
};
