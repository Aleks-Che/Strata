// Included in the same thread-confined backend translation unit as sync_runtime.
// Reuse the already-tested bounded cache policy; MiniMax owns its identities,
// copy completion, model lifetime and workspace/admission policy.
#include "../step35/expert_cache.hpp"
#include "gpu_arena.hpp"
#include <array>
static thread_local std::unique_ptr<step35::ExpertCache> mm27_cache;
static thread_local std::shared_ptr<minimax_m2::GpuArena> mm27_arena;
static thread_local uint64_t mm27_generation=0;
static thread_local bool mm27_cache_admit=false;
static thread_local const void *mm27_bound_model=nullptr;
struct mm27_tensor_identity {
    uint32_t id;
    std::string name;
    ggml_type type;
    const void *data;
    std::array<int64_t,4> ne;
    std::array<size_t,4> nb;
    std::shared_ptr<strata_expert_file::Source> file;
    uint64_t offset;
};
static thread_local std::map<const ggml_tensor *,mm27_tensor_identity> mm27_tensors;
void strata_mm27_cache_clear() {
    // All copies and graph consumers have completed before any public boundary.
    if(mm27_cache)mm27_cache->trim(0);
    mm27_cache_admit=false;
}
void strata_mm27_cache_configure(uint64_t bytes,bool use_arena) {
    mm27_cache.reset();mm27_arena.reset();mm27_cache_admit=false;
    if(!bytes)return;
    if(mm27_mode!=2)throw std::runtime_error("MiniMax GPU cache requires file-copy mode 2");
    const auto probe=[] {
        auto sample=step35::memory_sample();
        sample.gpu_free=std::min<uint64_t>(sample.gpu_free,mm27_test_limits.vram_available);
        sample.ram_free=std::min<uint64_t>(sample.ram_free,mm27_test_limits.ram_available);
        return sample;
    };
    if(use_arena)mm27_arena=std::make_shared<minimax_m2::GpuArena>(size_t(bytes),[probe] {
        const auto m=probe();MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
        if(!GlobalMemoryStatusEx(&ram))throw std::runtime_error("MiniMax commit sample unavailable");
        return minimax_m2::GpuArena::Memory{m.gpu_free,m.gpu_total,size_t(ram.ullAvailPageFile)};
    });
    const auto arena=mm27_arena;
    mm27_cache=std::make_unique<step35::ExpertCache>(size_t(bytes),probe,[arena](void **p,size_t bytes) {
        // WDDM device allocations also consume host commit. Refuse growth when
        // commit headroom is low, even if physical RAM remains available.
        MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
        if(!GlobalMemoryStatusEx(&ram))return cudaErrorUnknown;
        if(mm27_test_limits.cache_oom || ram.ullAvailPageFile<(1ull<<30) ||
            bytes>ram.ullAvailPageFile-(1ull<<30))return cudaErrorMemoryAllocation;
        return arena?arena->allocate(p,bytes):cudaMalloc(p,bytes);
    },[arena](void *p){return arena?arena->release(p):cudaFree(p);});
    if(arena)mm27_cache->set_backing_bytes([arena]{return arena->snapshot().reserved;});
    mm27_cache->set_reuse_allocations(true);
    mm27_cache->set_match_size(true);
    mm27_cache->set_fast_scan(true);
}
static void mm27_cache_refresh() {
    if(!mm27_cache)return;
    mm27_cache->refresh();
    if(!mm27_arena)return;
    // Logical eviction alone may leave partially occupied physical slabs.
    // Continue reclaiming completed slots until backing, not merely live bytes,
    // fits the sampled physical budget. Plan/pending-fill pins protect every
    // asynchronous user; inability to reclaim fails closed and drains the plan.
    const auto limit=mm27_cache->budget();
    while(mm27_arena->snapshot().reserved>limit) {
        const auto s=mm27_arena->snapshot();
        const auto excess=s.reserved-limit;
        const size_t remove=std::max<size_t>(excess,64u<<20);
        mm27_cache->trim(s.live>remove?s.live-remove:0);
        if(mm27_arena->snapshot().live==s.live && mm27_arena->snapshot().reserved>limit)
            throw std::runtime_error("MiniMax pinned cache cannot shrink under memory pressure; drain plan first");
        if(mm27_arena->snapshot().reserved>limit && !mm27_cache->resident_bytes())
            throw std::runtime_error("MiniMax arena retained empty backing");
    }
}
void strata_mm27_cache_bind(const void *model,const ggml_tensor *const *tensors,size_t count) {
    strata_mm27_cache_clear();mm27_tensors.clear();mm27_bound_model=nullptr;
    if(++mm27_generation==0)throw std::runtime_error("MiniMax cache generation overflow");
    for(size_t i=0;i<count;++i) {
        const auto *t=tensors[i];
        auto file=strata_expert_file::find(t->data,ggml_nbytes(t));
        if(!file || !ggml_is_contiguous(t) || t->ne[3]!=1)
            throw std::runtime_error("invalid MiniMax cache tensor mapping");
        mm27_tensor_identity item{uint32_t(i),t->name,t->type,t->data,{}, {},file,file->offset_of(t->data,ggml_nbytes(t))};
        std::copy(t->ne,t->ne+4,item.ne.begin());std::copy(t->nb,t->nb+4,item.nb.begin());
        mm27_tensors.emplace(t,std::move(item));
    }
    mm27_bound_model=model;
}
void strata_mm27_cache_unbind(const void *model) {
    if(mm27_bound_model!=model)return;
    strata_mm27_cache_clear();mm27_tensors.clear();mm27_bound_model=nullptr;
}
void strata_mm27_cache_decode(bool admit) {mm27_cache_admit=admit;}
static uint32_t mm27_tensor_id(const ggml_tensor *src) {
    const auto it=mm27_tensors.find(src);
    if(!mm27_bound_model || it==mm27_tensors.end())throw std::runtime_error("unbound MiniMax cached tensor");
    const auto &t=it->second;
    const auto file=strata_expert_file::find(src->data,ggml_nbytes(src));
    if(t.name!=src->name || t.type!=src->type || t.data!=src->data ||
        !std::equal(t.ne.begin(),t.ne.end(),src->ne) || !std::equal(t.nb.begin(),t.nb.end(),src->nb) ||
        file!=t.file || !file || file->offset_of(src->data,ggml_nbytes(src))!=t.offset)
        throw std::runtime_error("MiniMax cached tensor identity changed");
    return t.id;
}
static void mm27_cache_snapshot(strata_mm27_stats &s) {
    if(!mm27_cache)return;
    const auto c=mm27_cache->counters();
    s.cache_hits=c.hits;s.cache_misses=c.misses;s.cache_evictions=c.evictions;s.cache_reuses=c.reuses;
    s.cache_oom=c.oom;s.cache_bypasses=c.bypasses+c.rejected;
    s.cache_resident=mm27_cache->resident_bytes();s.cache_limit=mm27_cache->budget();
    if(mm27_arena) {
        const auto a=mm27_arena->snapshot();
        s.arena_reserved=a.reserved;s.arena_live=a.live;s.arena_blocks=a.blocks;
        s.arena_allocations=a.allocations;s.arena_frees=a.frees;s.arena_rejects=a.budget_rejects;
    }
}
