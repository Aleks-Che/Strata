#pragma once
#include "../step35/expert_cache.hpp"
#include <array>
#include <limits>
#include <set>

namespace minimax_m2 {
// MiniMax-only adapter. The bounded policy owns one allocation and frequency
// entry per expert triplet; callers still transfer individual tensor slices.
// Readiness is published only after the caller's CUDA completion fence.
class ExpertCache : public step35::ExpertCache {
    using Base=step35::ExpertCache;
    using Key=step35::MatrixKey;
public:
    using Base::Base;
    struct Triplet {std::array<uint32_t,3> tensors;std::array<size_t,3> strides;uint32_t experts;};
    struct GroupStats {uint64_t groups=0,partial=0,ready_matrices=0,pending_matrices=0,ready_bytes=0,admissions=0;};
private:
    struct Layout {Triplet spec;std::array<size_t,3> offsets;size_t bytes;};
    struct Part {uint32_t group=0,role=0;bool valid=false;};
    struct State {uint8_t ready=0,pending=0;};
    std::vector<Layout> layouts;
    std::vector<Part> parts;
    std::map<Key,State> states;
    uint64_t hits=0,misses=0,admissions=0,skipped=0;
    bool plan_active=false;
    std::set<Key> trained,attempted;
    static size_t aligned(size_t bytes) {
        if(bytes>std::numeric_limits<size_t>::max()-65535)throw std::runtime_error("MiniMax group size overflow");
        return (bytes+65535)/65536*65536;
    }
    const Part &part(const Key &k) const {
        if(k.tensor>=parts.size() || !parts[k.tensor].valid || k.expert>=layouts[parts[k.tensor].group].spec.experts)
            throw std::runtime_error("MiniMax group key outside registered tensors");
        return parts[k.tensor];
    }
    Key group_key(const Key &k) const {return {k.generation,part(k).group,k.expert};}
    size_t stored(const Layout &l,uint32_t role,uint32_t expert) const {
        return l.spec.strides[role]+std::min<size_t>(512,l.spec.strides[role]*(l.spec.experts-1-expert));
    }
    const Part &validate(const Key &k,size_t bytes) const {
        const auto &p=part(k);
        if(stored(layouts[p.group],p.role,k.expert)!=bytes)throw std::runtime_error("MiniMax group matrix size changed");
        return p;
    }
public:
    bool grouped() const {return !layouts.empty();}
    void begin_plan() {
        if(!grouped())return;
        if(plan_active)throw std::runtime_error("MiniMax group plan already active");
        plan_active=true;trained.clear();attempted.clear();
    }
    void end_plan() {plan_active=false;trained.clear();attempted.clear();}
    void set_groups(size_t tensor_count,const std::vector<Triplet> &specs) {
        if(plan_active || Base::size() || !tensor_count || specs.size()!=tensor_count/3 || tensor_count%3)
            throw std::runtime_error("MiniMax groups require an empty cache and complete triplets");
        std::vector<Layout> next;std::vector<Part> index(tensor_count);
        for(const auto &spec:specs) {
            if(!spec.experts || spec.experts>256)throw std::runtime_error("invalid MiniMax group expert count");
            Layout l{spec,{},0};
            for(uint32_t role=0;role<3;++role) {
                const auto id=spec.tensors[role];const auto stride=spec.strides[role];
                if(id>=tensor_count || index[id].valid || !stride || stride>std::numeric_limits<size_t>::max()-512 ||
                   stride>std::numeric_limits<size_t>::max()/spec.experts)
                    throw std::runtime_error("invalid MiniMax group tensor layout");
                index[id]={uint32_t(next.size()),role,true};l.offsets[role]=l.bytes;
                const auto n=aligned(stored(l,role,0));
                if(l.bytes>std::numeric_limits<size_t>::max()-n)throw std::runtime_error("MiniMax group layout overflow");
                l.bytes+=n;
            }
            next.push_back(l);
        }
        layouts=std::move(next);parts=std::move(index);states.clear();
    }
    bool contains(const Key &k,size_t bytes) const {
        if(!grouped())return Base::contains(k,bytes);
        const auto &p=validate(k,bytes);const auto g=group_key(k);const auto it=states.find(g);
        return Base::contains(g,layouts[p.group].bytes) && it!=states.end() && (it->second.ready&(1u<<p.role));
    }
    void *get(const Key &k,size_t bytes,bool train=true) {
        if(!grouped())return Base::get(k,bytes,train);
        const auto &p=validate(k,bytes);const auto g=group_key(k);
        // One observed route trains one group, not three progressively hotter
        // matrix requests. Otherwise admission can first succeed at DOWN.
        const bool record=train && (!plan_active || trained.insert(g).second);
        void *base=Base::get(g,layouts[p.group].bytes,record);const auto it=states.find(g);
        if(!base || it==states.end() || !(it->second.ready&(1u<<p.role))) {++misses;return nullptr;}
        ++hits;return static_cast<char *>(base)+layouts[p.group].offsets[p.role];
    }
    void *admit(const Key &k,size_t bytes) {
        if(!grouped())return Base::admit(k,bytes);
        const auto &p=validate(k,bytes);const auto g=group_key(k);const auto &l=layouts[p.group];
        void *base=Base::get(g,l.bytes,false);
        if(!base) {
            if(plan_active && !attempted.insert(g).second) {++skipped;return nullptr;}
            base=Base::admit(g,l.bytes);if(!base)return nullptr;
            // The same pointer can be reused for a different expert. Reset
            // readiness on EVERY group admission, even when its address repeats.
            states[g]={};++admissions;
        }
        auto &s=states.at(g);const auto bit=uint8_t(1u<<p.role);
        if((s.ready|s.pending)&bit)throw std::runtime_error("duplicate MiniMax group matrix admission");
        s.pending|=bit;return static_cast<char *>(base)+l.offsets[p.role];
    }
    void commit(const Key &k,size_t bytes) {
        if(!grouped())return;
        const auto &p=validate(k,bytes);const auto g=group_key(k);const auto bit=uint8_t(1u<<p.role);
        const auto it=states.find(g);
        if(!Base::contains(g,layouts[p.group].bytes) || it==states.end() || !(it->second.pending&bit))
            throw std::runtime_error("MiniMax group fill lost before completion");
        it->second.pending&=uint8_t(~bit);it->second.ready|=bit;
    }
    std::unique_ptr<PlanPins> protect(const std::vector<Key> &keys) {
        if(!grouped())return Base::protect(keys);
        std::vector<Key> groups;groups.reserve(keys.size());
        for(const auto &k:keys)groups.push_back(group_key(k));
        std::sort(groups.begin(),groups.end());groups.erase(std::unique(groups.begin(),groups.end()),groups.end());
        // Include incomplete groups: a pending fill owns the whole allocation.
        return Base::protect(groups);
    }
    void trim(size_t bytes) {
        Base::trim(bytes);if(!Base::size())states.clear();
    }
    Counters counters() const {
        auto c=Base::counters();if(grouped()){c.hits=hits;c.misses=misses;c.bypasses+=skipped;}return c;
    }
    void reset_counters() {Base::reset_counters();hits=misses=admissions=skipped=0;}
    GroupStats group_stats() const {
        GroupStats s;s.admissions=admissions;
        for(const auto &item:states) {
            const auto &l=layouts.at(item.first.tensor);if(!Base::contains(item.first,l.bytes))continue;
            ++s.groups;s.partial+=item.second.ready!=7;
            for(uint32_t role=0;role<3;++role) {
                const bool ready=(item.second.ready&(1u<<role))!=0;
                s.ready_matrices+=ready;s.pending_matrices+=(item.second.pending&(1u<<role))!=0;
                if(ready)s.ready_bytes+=stored(l,role,item.first.expert);
            }
        }
        return s;
    }
};
} // namespace minimax_m2
