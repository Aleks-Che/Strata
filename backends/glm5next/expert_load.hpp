#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata_glm {
inline bool ram_experts() {
    const auto *value=std::getenv("STRATA_GLM_EXPERT_LOAD_RAM");
    if(!value)return false;
    if(std::strcmp(value,"0") && std::strcmp(value,"1"))
        throw std::runtime_error("STRATA_GLM_EXPERT_LOAD_RAM must be 0 or 1");
    return *value=='1';
}
// -1 means all routed expert layers; an explicit bound keeps the remainder mapped.
inline int ram_expert_layers() {
    const auto *value=std::getenv("STRATA_GLM_EXPERT_RAM_LAYERS");
    if(!value)return -1;
    if(!ram_experts() || !*value)throw std::runtime_error("RAM layers require STRATA_GLM_EXPERT_LOAD_RAM=1");
    int layers=0;
    for(auto p=value;*p;++p) {
        if(*p<'0' || *p>'9' || layers>4096)throw std::runtime_error("STRATA_GLM_EXPERT_RAM_LAYERS must be 0..4096");
        layers=layers*10+(*p-'0');
    }
    if(layers>4096)throw std::runtime_error("STRATA_GLM_EXPERT_RAM_LAYERS must be 0..4096");
    return layers;
}
inline bool mapped_expert_layer(int layer) {
    return !ram_experts() || (ram_expert_layers()>=0 && layer>=ram_expert_layers());
}
inline const char *expert_storage_name() {
    return !ram_experts()?"mmap":ram_expert_layers()<0?"ram":"hybrid";
}
}
