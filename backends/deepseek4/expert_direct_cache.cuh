#pragma once
#include <array>
#include <unordered_set>
#include <vector>

struct StrataExpertDirect {
    static constexpr size_t max_experts=256;
    struct Packet { const void *pointers[max_experts]; } packet{};
    struct Held { const void *key,*source; void *destination; size_t bytes; };
    std::vector<Held> held;
    std::unordered_set<const void *> keys;
    std::vector<cudaStream_t> readers;
    StrataExpertReuseFence fence;
    const ggml_tensor *tensor=nullptr;
    void *data=nullptr;
    size_t width=0,count=0;
    const void **table=nullptr;
    bool enabled=false,dirty=false,consumed=false;

    StrataExpertDirect() {
        const char *v=std::getenv("STRATA_DS4_DIRECT_CACHE");
        enabled=v && std::strcmp(v,"1")==0;
#ifndef STRATA_DS4_DIRECT_CACHE_COMPILED
        if(enabled)GGML_ABORT("STRATA_DS4_DIRECT_CACHE requires STRATA_DEEPSEEK_DIRECT_CACHE_EXPERIMENT=ON");
#endif
    }
    ~StrataExpertDirect(){if(table)CUDA_CHECK(cudaFree(table));}
    void prepare(const ggml_tensor *weight) {
        GGML_ASSERT(!tensor && held.empty());
        if(!table)CUDA_CHECK(cudaMalloc((void **)&table,sizeof(Packet)));
        tensor=weight;data=weight->data;width=weight->nb[2];count=size_t(weight->ne[2]);
        consumed=false;readers.clear();fallback_table();
    }
    void fallback_table() {
        for(size_t i=0;i<max_experts;++i)packet.pointers[i]=i<count?(uint8_t *)data+i*width:nullptr;
        dirty=true;
    }
    bool bind(const ggml_tensor *weight,int id,const void *key,const void *source,size_t bytes) {
        if(!tensor || tensor!=weight || data!=weight->data)return false;
        GGML_ASSERT(!consumed && id>=0 && size_t(id)<count);
        packet.pointers[id]=source;keys.insert(key);
        held.push_back({key,source,(uint8_t *)data+size_t(id)*width,bytes});dirty=true;return true;
    }
    void reader(cudaStream_t stream) {
        if(std::find(readers.begin(),readers.end(),stream)==readers.end())readers.push_back(stream);
    }
    void complete(cudaStream_t owner,cudaStream_t last) {
        if(!tensor)return;
        // Captured graph replay is ordered on the owner stream. For ordinary
        // multi-stream evaluation, join every stream that obtained the table.
        reader(last);for(auto stream:readers)fence.order(stream,owner);
        readers.clear();consumed=true;
    }
    void clear() {tensor=nullptr;data=nullptr;held.clear();keys.clear();readers.clear();dirty=false;consumed=false;}
    void publish(cudaStream_t stream);
};

static __global__ void strata_direct_publish(const void **table,StrataExpertDirect::Packet packet) {
    table[threadIdx.x]=packet.pointers[threadIdx.x];
}
inline void StrataExpertDirect::publish(cudaStream_t stream) {
    if(!dirty)return;
    // Kernel arguments are copied by value before returning to the caller.
    // No asynchronous read from a reused pageable or pinned host array.
    strata_direct_publish<<<1,max_experts,0,stream>>>(table,packet);
    CUDA_CHECK(cudaGetLastError());dirty=false;
}
