#pragma once
#include "ggml.h"
#include <cstring>
#include <vector>

// One owned snapshot, valid for one selected-copy split and one exact node.
// No GPU addresses are used as keys across splits, graphs or model lifetimes.
namespace minimax_m2 {
class RouterIds {
    const ggml_tensor *node_=nullptr, *ids_=nullptr, *weight_=nullptr;
    ggml_tensor identity_{};
    int64_t experts_=0;
    std::vector<char> bytes_;
public:
    void clear() {node_=nullptr;ids_=nullptr;weight_=nullptr;}
    bool publish(const ggml_tensor *node,const ggml_tensor *source,const void *data,size_t bytes) {
        clear();
        if(!node || node->op!=GGML_OP_MUL_MAT_ID || !node->src[0] || !node->src[2] || !source || !data)return false;
        const auto *ids=node->src[2];
        if(ids->type!=GGML_TYPE_I32 || source->type!=GGML_TYPE_I32 || !ids->data ||
            ids->ne[0]<=0 || ids->ne[1]<=0 || ids->ne[2]!=1 || ids->ne[3]!=1 ||
            bytes==0 || bytes>65536 || bytes!=ggml_nbytes(ids) || bytes!=ggml_nbytes(source) ||
            std::memcmp(ids->ne,source->ne,sizeof(ids->ne)) || std::memcmp(ids->nb,source->nb,sizeof(ids->nb)))return false;
        for(int64_t t=0;t<ids->ne[1];++t)for(int64_t e=0;e<ids->ne[0];++e) {
            const size_t offset=t*ids->nb[1]+e*ids->nb[0];
            if(offset>bytes-sizeof(int32_t))return false;
            int32_t id;std::memcpy(&id,static_cast<const char *>(data)+offset,sizeof(id));
            if(id<0 || id>=node->src[0]->ne[2])return false;
        }
        bytes_.resize(bytes);std::memcpy(bytes_.data(),data,bytes);
        identity_=*ids;ids_=ids;weight_=node->src[0];experts_=weight_->ne[2];node_=node;
        return true;
    }
    bool take(const ggml_tensor *node,void *data,size_t bytes) {
        const bool match=node && node==node_ && node->src[0]==weight_ && node->src[2]==ids_ &&
            weight_->ne[2]==experts_ && ids_->type==identity_.type && ids_->data==identity_.data &&
            ids_->buffer==identity_.buffer && ids_->view_src==identity_.view_src && ids_->view_offs==identity_.view_offs &&
            !std::memcmp(ids_->ne,identity_.ne,sizeof(ids_->ne)) && !std::memcmp(ids_->nb,identity_.nb,sizeof(ids_->nb)) &&
            bytes==bytes_.size() && data;
        clear(); // consume once, including mismatches (sliced/tokenwise nodes fall back).
        if(match)std::memcpy(data,bytes_.data(),bytes);
        return match;
    }
};
struct RouterIdsScope {
    RouterIds &ids;
    explicit RouterIdsScope(RouterIds &slot):ids(slot) {ids.clear();}
    ~RouterIdsScope() {ids.clear();}
    RouterIdsScope(const RouterIdsScope &)=delete;
    RouterIdsScope &operator=(const RouterIdsScope &)=delete;
};
}
// Defined in MiniMax's private ggml-base; called on the scheduler's thread
// during CUDA graph submission. CUDA graph recording remains disabled.
bool strata_mm27_router_ids_take(const ggml_tensor *node,void *data,size_t bytes);
