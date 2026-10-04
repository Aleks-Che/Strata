#pragma once
#include "expert_cache.hpp"
#include "expert_transport.hpp"
#include <optional>

namespace strata_glm {
// One ordered selected-expert copy scope. cache and transport must outlive this
// object and must not be mutated externally while it is active. Destinations and
// streams outlive their queued GPU consumers. No model graph or compute here.
class ExpertDispatch {
    ExpertCache &cache;
    ExpertTransport &transport;
    std::vector<ExpertKey> keys;
    std::vector<ExpertSourceView> sources;
    std::optional<ExpertCache::PlanPins> pins;
    std::vector<ExpertCache::Lease> leases;
    std::vector<bool> hits;
    size_t next=0,miss=0;
    bool active=false;
    static void check(cudaError_t error) {
        if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
    }
    void release() noexcept {
        leases.clear();pins.reset();sources.clear();active=false;
    }
    void end(bool complete) {
        if(!active)return;
        try {if(complete)transport.finish();else transport.cancel();}
        catch(...) {release();throw;}
        release();
    }
    void cancel_noexcept() noexcept {try {end(false);}catch(...) {}}
    std::shared_ptr<const void> source_owner(const ExpertKey &key) const {
        for(const auto &source:sources)
            if(source.model==key.model && source.generation==key.generation && source.shard==key.shard)
                return source.owner;
        return {};
    }
public:
    ExpertDispatch(ExpertCache &c,ExpertTransport &t,std::vector<ExpertKey> plan,
                   std::vector<ExpertSourceView> views,bool decode)
        :cache(c),transport(t),keys(std::move(plan)),sources(std::move(views)) {
        // Reject nested scopes before taking pins or touching transport state.
        if(transport.in_progress())throw std::logic_error("GLM dispatch transport already active");
        std::set<ExpertKey> unique;
        for(const auto &key:keys)
            if(!unique.insert(key).second)throw std::invalid_argument("duplicate GLM dispatch key");
        pins.emplace(cache.protect_plan(keys));
        std::vector<ExpertKey> misses;
        for(const auto &key:keys) {
            const bool hit=cache.resident(key);hits.push_back(hit);
            if(!hit)misses.push_back(key);
        }
        // Validation and source retention happen before any transfer. Only
        // misses are prefetched; all-hit plans need no source mappings.
        transport.begin(std::move(misses),sources,decode);active=true;
    }
    ExpertDispatch(const ExpertDispatch&)=delete;
    ExpertDispatch &operator=(const ExpertDispatch&)=delete;
    ~ExpertDispatch() {cancel_noexcept();}
    size_t remaining() const {return active?keys.size()-next:0;}
    void copy(size_t index,void *destination,cudaStream_t stream) {
        if(!active || index!=next || index>=keys.size() || !destination)
            throw std::logic_error("GLM dispatch requires the next matrix and destination");
        try {
            const auto &key=keys[index];
            if(cache.resident(key)!=hits[index])
                throw std::logic_error("GLM cache residency changed during dispatch");
            auto lease=cache.get(key,source_owner(key),stream,[&](void *data,size_t,cudaStream_t s) {
                if(hits[index])throw std::logic_error("GLM planned hit unexpectedly uploaded");
                transport.transfer(miss,data,s);
            });
            if(lease) {
                check(cudaMemcpyAsync(destination,lease.data(),size_t(key.bytes),cudaMemcpyDeviceToDevice,stream));
                leases.push_back(std::move(lease));
            }else {
                if(hits[index])throw std::logic_error("GLM planned hit unexpectedly bypassed");
                transport.transfer(miss,destination,stream);
            }
            if(!hits[index])++miss;
            ++next;
        }catch(...) {cancel_noexcept();throw;}
    }
    void finish() {
        if(active && next!=keys.size()) {
            end(false);throw std::logic_error("incomplete GLM dispatch; remainder cancelled");
        }
        end(true);
    }
    void cancel() {end(false);}
};
}
