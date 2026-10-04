#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

namespace strata_glm {
enum class Branch { main, mtp };
enum class Projection { gate, up, down };

// Native equivalent of ExpertMatrix.cache_key in tools/glm5next_expert_plan.py.
// Identity refers to the complete model, not its architecture or mmap address.
struct ExpertKey {
    std::string model;
    uint64_t generation;
    Branch branch;
    int layer,expert;
    Projection projection;
    std::string quant;
    uint64_t columns,rows;
    std::string shard;
    uint64_t offset,bytes;
    auto fields() const {
        return std::tie(model,generation,branch,layer,expert,projection,quant,
                        columns,rows,shard,offset,bytes);
    }
    bool operator<(const ExpertKey &other) const {return fields()<other.fields();}
    bool operator==(const ExpertKey &other) const {return fields()==other.fields();}
    void validate() const {
        if(model.empty() || shard.empty() || quant.empty() || layer<0 || expert<0 ||
           (branch!=Branch::main && branch!=Branch::mtp) ||
           (projection!=Projection::gate && projection!=Projection::up && projection!=Projection::down) ||
           !columns || !rows || !bytes || bytes>std::numeric_limits<size_t>::max() ||
           offset>std::numeric_limits<uint64_t>::max()-bytes)
            throw std::invalid_argument("invalid GLM expert cache key");
    }
};

struct ExpertKeyHash {
    size_t operator()(const ExpertKey &key) const {
        size_t seed=0;
        std::apply([&](const auto &...fields) {
            auto combine=[&](const auto &value) {
                seed^=std::hash<std::decay_t<decltype(value)>>{}(value)+size_t(0x9e3779b9)+(seed<<6)+(seed>>2);
            };
            (combine(fields),...);
        },key.fields());
        return seed;
    }
};

}
