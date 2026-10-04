#pragma once
#include "expert_key.hpp"
#include <array>
#include <climits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace strata_glm {
// Loader-supplied GGUF geometry, in bytes and [columns, rows, experts] order.
// This planner reads no payload and owns no mapping; consumers retain sources.
struct ExpertShardLayout {
    std::string name;
    uint64_t data_start,file_bytes;
};
struct ExpertTensorLayout {
    Projection projection;
    std::string quant;
    uint64_t columns,rows,experts;
    size_t shard;
    uint64_t offset,bytes;
};
struct ExpertLayerLayout {
    int layer,main_blocks,block_count,leading_dense_blocks;
    std::array<ExpertTensorLayout,3> tensors;
};

inline uint64_t expert_checked_product(uint64_t a,uint64_t b) {
    if(b && a>UINT64_MAX/b)throw std::invalid_argument("GLM expert layout size overflow");
    return a*b;
}
inline uint64_t expert_block_bytes(const std::string &quant) {
    // All eight routed formats observed in the two GLM profiles use QK=256.
    if(quant=="IQ2_S")return 82;
    if(quant=="IQ3_S" || quant=="Q3_K")return 110;
    if(quant=="IQ4_XS")return 136;
    if(quant=="Q2_K")return 84;
    if(quant=="IQ3_XXS")return 98;
    if(quant=="Q6_K")return 210;
    if(quant=="Q4_K")return 144;
    throw std::invalid_argument("unsupported GLM routed quant: "+quant);
}

// Batch routes may repeat experts. Preserve first-use order and every selected
// expert's complete gate/up/down triple; do not pad between expert matrices.
// max_routes bounds input processing and result metadata, not payload memory.
inline std::vector<ExpertKey> plan_experts(
        const std::string &model,uint64_t generation,const ExpertLayerLayout &layer,
        const std::vector<ExpertShardLayout> &shards,const std::vector<int> &routes,
        size_t max_routes=4096) {
    if(model.empty() || layer.main_blocks<=0 || layer.block_count<layer.main_blocks ||
       layer.leading_dense_blocks<0 || layer.leading_dense_blocks>layer.main_blocks ||
       layer.layer<layer.leading_dense_blocks || layer.layer>=layer.block_count || !max_routes || routes.size()>max_routes)
        throw std::invalid_argument("invalid GLM layer identity or route limit");
    std::unordered_set<std::string> names;
    for(const auto &shard:shards)
        // A metadata-only shard may end before its aligned data_start.
        if(shard.name.empty() || !names.insert(shard.name).second)
            throw std::invalid_argument("invalid or duplicate GLM shard layout");
    std::array<const ExpertTensorLayout *,3> tensors{};
    std::array<uint64_t,3> matrix_bytes{};
    for(const auto &tensor:layer.tensors) {
        const auto p=static_cast<unsigned>(tensor.projection);
        if(p>=3 || tensors[p] || !tensor.columns || !tensor.rows ||
           !tensor.experts || tensor.experts>INT_MAX || tensor.columns%256 || tensor.shard>=shards.size())
            throw std::invalid_argument("invalid GLM expert projection, shape or shard");
        const auto bytes=expert_checked_product(
            expert_checked_product(tensor.columns/256,expert_block_bytes(tensor.quant)),tensor.rows);
        if(bytes>std::numeric_limits<size_t>::max() ||
           expert_checked_product(bytes,tensor.experts)!=tensor.bytes)
            throw std::invalid_argument("GLM expert tensor byte count disagrees with shape");
        const auto &shard=shards[tensor.shard];
        if(tensor.offset<shard.data_start || tensor.offset>shard.file_bytes || tensor.bytes>shard.file_bytes-tensor.offset)
            throw std::invalid_argument("GLM expert tensor outside shard payload");
        tensors[p]=&tensor;matrix_bytes[p]=bytes;
    }
    const auto &gate=*tensors[0],&up=*tensors[1],&down=*tensors[2];
    if(gate.columns!=up.columns || gate.rows!=up.rows || gate.experts!=up.experts ||
       gate.columns!=down.rows || gate.rows!=down.columns || gate.experts!=down.experts)
        throw std::invalid_argument("GLM gate/up/down layouts disagree");
    for(size_t p=0;p<3;++p)for(size_t q=p+1;q<3;++q) {
        const auto &a=*tensors[p],&b=*tensors[q];
        if(a.shard==b.shard && a.offset<b.offset+b.bytes && b.offset<a.offset+a.bytes)
            throw std::invalid_argument("overlapping GLM expert tensors");
    }
    std::unordered_set<int> seen;
    std::vector<ExpertKey> result;
    for(int expert:routes) {
        if(expert<0 || uint64_t(expert)>=gate.experts)
            throw std::invalid_argument("GLM router ID outside expert tensor");
        if(!seen.insert(expert).second)continue;
        for(size_t p=0;p<3;++p) {
            const auto &t=*tensors[p];
            ExpertKey key{model,generation,layer.layer<layer.main_blocks?Branch::main:Branch::mtp,
                          layer.layer,expert,t.projection,t.quant,t.columns,t.rows,shards[t.shard].name,
                          t.offset+uint64_t(expert)*matrix_bytes[p],matrix_bytes[p]};
            key.validate();result.push_back(std::move(key));
        }
    }
    return result;
}
}
