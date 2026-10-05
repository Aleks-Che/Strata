#pragma once
#include "expert_key.hpp"
#include "../common/expert_pipeline.hpp"
#include <map>
#include <memory>
#include <set>

namespace strata_glm {
// Complete offset-zero source view. owner must own the mapping/registration,
// not merely a file handle. The caller attests the full model/load identity.
struct ExpertSourceView {
    std::string model;
    uint64_t generation;
    std::string shard;
    const uint8_t *data;
    size_t bytes;
    std::shared_ptr<const void> owner;
};

// One host owner/device; cancellation is serialized with transfer calls. This
// wraps transport lifetime, not graph execution or cache admission. Destinations
// and their streams must outlive queued consumers, including after cancel().
class ExpertTransport {
    // Reverse destruction order is intentional: join pipeline workers and drain
    // GPU ring consumers before dropping the last mapping owners on exception.
    std::vector<std::shared_ptr<const void>> owners;
    std::vector<ExpertKey> keys;
    std::vector<StrataExpertSlice> slices;
    size_t next=0,chunk_bytes;
    bool active=false;
    std::unique_ptr<StrataExpertPipeline> pipeline;

    void clear_plan() noexcept {
        slices.clear();keys.clear();owners.clear();next=0;active=false;
    }
    void failed() noexcept {
        pipeline.reset(); // joins readers/drains GPU before releasing mappings
        clear_plan();
    }
    void end(bool require_complete) {
        if(!active)return;
        const bool incomplete=next!=keys.size();
        try {pipeline->finish();}catch(...) {failed();throw;}
        clear_plan();
        if(require_complete && incomplete)
            throw std::logic_error("incomplete GLM transfer plan; remainder cancelled");
    }
public:
    ExpertTransport(int gpu,size_t chunk,bool write_combined=false,int readers=2,int mode=0,StrataExpertPipeline::CopyObserver observer={},int decode_readers=1,bool early_host_refill=false)
        :chunk_bytes(chunk),pipeline(std::make_unique<StrataExpertPipeline>(gpu,chunk,write_combined,readers,mode,std::move(observer),decode_readers,early_host_refill)) {}
    ExpertTransport(const ExpertTransport&)=delete;
    ExpertTransport &operator=(const ExpertTransport&)=delete;
    // Default destruction drains the pipeline before destroying owners.
    bool in_progress() const {return active;}
    size_t remaining() const {return keys.size()-next;}
    StrataExpertPipeline::Counters counters() {
        if(!pipeline)throw std::logic_error("failed GLM transport must be recreated");
        return pipeline->counters();
    }
    void begin(std::vector<ExpertKey> plan,const std::vector<ExpertSourceView> &sources,bool decode) {
        if(!pipeline)throw std::logic_error("failed GLM transport must be recreated");
        if(active)throw std::logic_error("GLM transfer plan already active");
        if(plan.size()>12288)throw std::invalid_argument("GLM transfer plan exceeds matrix limit");
        using Identity=std::tuple<std::string,uint64_t,std::string>;
        std::map<Identity,const ExpertSourceView *> views;
        for(const auto &source:sources) {
            if(source.model.empty() || source.shard.empty() || !source.owner || !source.data || !source.bytes ||
               uintptr_t(source.data)>UINTPTR_MAX-source.bytes ||
               !views.emplace(Identity{source.model,source.generation,source.shard},&source).second)
                throw std::invalid_argument("invalid or duplicate GLM source view");
        }
        std::set<ExpertKey> unique;
        std::vector<StrataExpertSlice> pending;
        std::vector<std::shared_ptr<const void>> retained;
        std::set<Identity> retained_ids;
        size_t chunks=0;
        for(const auto &key:plan) {
            key.validate();
            if(!unique.insert(key).second)throw std::invalid_argument("duplicate GLM transfer key");
            Identity id{key.model,key.generation,key.shard};
            const auto found=views.find(id);
            if(found==views.end())throw std::invalid_argument("GLM plan has no matching source identity");
            const auto &source=*found->second;
            if(key.offset>source.bytes || key.bytes>source.bytes-key.offset)
                throw std::invalid_argument("GLM transfer range outside source view");
            // Subtraction/division avoid an overflowing ceil(bytes/chunk).
            const size_t count=1+(size_t(key.bytes)-1)/chunk_bytes;
            if(count>1048576-chunks)throw std::invalid_argument("GLM transfer plan exceeds chunk limit");
            chunks+=count;
            pending.push_back({source.data+size_t(key.offset),size_t(key.bytes),decode});
            if(retained_ids.insert(id).second)retained.push_back(source.owner);
        }
        // No jobs are submitted until the entire plan/source binding validates.
        owners=std::move(retained);keys=std::move(plan);slices=std::move(pending);next=0;active=true;
        try {pipeline->start(slices);}catch(...) {failed();throw;}
    }
    void transfer(size_t index,void *destination,cudaStream_t consumer) {
        if(!active || index!=next || index>=slices.size() || !destination)
            throw std::logic_error("GLM transfer requires the next planned matrix and a destination");
        try {
            const auto &slice=slices[index];
            if(!pipeline->transfer(destination,slice.data,slice.bytes,consumer))
                throw std::runtime_error("GLM transport lost its ordered pipeline plan");
            ++next;
        }catch(...) {failed();throw;}
    }
    void finish() {end(true);}
    void cancel() {end(false);}
};
}
