// Included in the same thread-confined backend translation unit as sync_runtime.
// Reuse the already-tested bounded cache policy; MiniMax owns its identities,
// copy completion, model lifetime and workspace/admission policy.
#include "../step35/expert_cache.hpp"
#include "gpu_arena.hpp"
#include "group_cache.hpp"
#include "host_cache.hpp"
#include <array>
static thread_local std::unique_ptr<minimax_m2::ExpertCache> mm27_cache;
static thread_local bool mm27_group_experts=false;
static thread_local std::shared_ptr<minimax_m2::GpuArena> mm27_arena;
static thread_local uint64_t mm27_generation=0;
static thread_local bool mm27_cache_admit=false;
static thread_local std::shared_ptr<minimax_m2::HostCache> mm27_ram_cache;
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
static void mm27_cache_setup_groups() {
    if(!mm27_cache || !mm27_group_experts || !mm27_bound_model)return;
    std::map<unsigned,std::array<const mm27_tensor_identity *,3>> layers;
    for(const auto &entry:mm27_tensors) {
        const auto &t=entry.second;const auto dot=t.name.find('.',4);
        if(t.name.rfind("blk.",0)!=0 || dot==std::string::npos || dot==4 ||
           t.name.substr(4,dot-4).find_first_not_of("0123456789")!=std::string::npos)
            throw std::runtime_error("invalid MiniMax grouped tensor name");
        const auto suffix=t.name.substr(dot);int role=-1;
        if(suffix==".ffn_gate_exps.weight")role=0;
        if(suffix==".ffn_up_exps.weight")role=1;
        if(suffix==".ffn_down_exps.weight")role=2;
        if(role<0)throw std::runtime_error("unsupported MiniMax grouped tensor role");
        const auto layer=std::stoul(t.name.substr(4,dot-4));
        if(layer>=256 || layers[unsigned(layer)][role])throw std::runtime_error("duplicate/invalid MiniMax grouped layer");
        layers[unsigned(layer)][role]=&t;
    }
    std::vector<minimax_m2::ExpertCache::Triplet> groups;
    for(const auto &entry:layers) {
        const auto &ts=entry.second;
        if(!ts[0] || !ts[1] || !ts[2] || ts[0]->ne[2]!=ts[1]->ne[2] || ts[0]->ne[2]!=ts[2]->ne[2])
            throw std::runtime_error("incomplete MiniMax expert group");
        groups.push_back({{ts[0]->id,ts[1]->id,ts[2]->id},{ts[0]->nb[2],ts[1]->nb[2],ts[2]->nb[2]},uint32_t(ts[0]->ne[2])});
    }
    mm27_cache->set_groups(mm27_tensors.size(),groups);
}
void strata_mm27_cache_clear() {
    // All copies and graph consumers have completed before any public boundary.
    if(mm27_cache)mm27_cache->trim(0);
    if(mm27_ram_cache) {mm27_ram_cache->clear();mm27_ram_cache->set_prefill(true);}
    mm27_cache_admit=false;
}
void strata_mm27_cache_configure(uint64_t bytes,bool use_arena,uint32_t block_mib,uint32_t growth_reserve_mib,bool group_experts) {
    if(block_mib!=8 && block_mib!=16 && block_mib!=32 && block_mib!=64)
        throw std::runtime_error("MiniMax arena block MiB must be 8, 16, 32 or 64");
    if(growth_reserve_mib>1024)throw std::runtime_error("MiniMax arena growth reserve MiB must be 0..1024");
    if(group_experts && (!bytes || !use_arena))throw std::runtime_error("MiniMax expert groups require a nonzero arena cache");
    mm27_cache.reset();mm27_arena.reset();mm27_cache_admit=false;mm27_group_experts=group_experts;
    if(mm27_ram_cache) {mm27_ram_cache->clear();mm27_ram_cache->set_gpu_partition(bytes>0);}
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
    },size_t(block_mib)<<20,size_t(growth_reserve_mib)<<20);
    const auto arena=mm27_arena;
    mm27_cache=std::make_unique<minimax_m2::ExpertCache>(size_t(bytes),probe,[arena](void **p,size_t bytes) {
        // WDDM device allocations also consume host commit. Refuse growth when
        // commit headroom is low, even if physical RAM remains available.
        MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
        if(!GlobalMemoryStatusEx(&ram))return cudaErrorUnknown;
        if(mm27_test_limits.cache_oom || ram.ullAvailPageFile<(1ull<<30) ||
            bytes>ram.ullAvailPageFile-(1ull<<30))return cudaErrorMemoryAllocation;
        return arena?arena->allocate(p,bytes):cudaMalloc(p,bytes);
    },[arena](void *p){return arena?arena->release(p):cudaFree(p);});
    if(arena) {
        mm27_cache->set_backing_bytes([arena]{return arena->snapshot().reserved;});
        mm27_cache->set_backing_groups([arena](const void *p){return arena->allocation_group(p);});
    }
    mm27_cache->set_reuse_allocations(true);
    mm27_cache->set_match_size(true);
    mm27_cache->set_fast_scan(true);
    mm27_cache_setup_groups();
}
static void mm27_cache_refresh() {
    if(!mm27_cache)return;
    mm27_cache->refresh();
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
    mm27_cache_setup_groups();
}
void strata_mm27_cache_unbind(const void *model) {
    if(mm27_bound_model!=model)return;
    strata_mm27_cache_clear();mm27_tensors.clear();mm27_bound_model=nullptr;
}
void strata_mm27_cache_decode(bool admit) {
    mm27_cache_admit=admit;if(mm27_ram_cache)mm27_ram_cache->set_prefill(!admit);
}
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
    if(mm27_ram_cache) {
        const auto h=mm27_ram_cache->snapshot();
        s.ram_cache_bytes=h.bytes;s.ram_cache_budget=h.budget;s.ram_cache_cap=h.cap;s.ram_cache_entries=h.entries;
        s.ram_cache_readers=h.readers;s.ram_cache_peak_bytes=h.peak_bytes;
        s.ram_cache_hits=h.hits;s.ram_cache_misses=h.misses;s.ram_cache_mapped_bytes=h.mapped_bytes;s.ram_cache_file_bytes=h.file_bytes;
        s.ram_cache_admissions=h.admissions;s.ram_cache_evictions=h.evictions;s.ram_cache_gpu_drops=h.gpu_drops;
        s.ram_cache_gpu_drop_bytes=h.gpu_drop_bytes;s.ram_cache_prefill_bypasses=h.prefill_bypasses;s.ram_cache_rejected=h.rejected;
        s.ram_cache_gpu_waits=h.gpu_waits;s.ram_cache_history_entries=h.history_entries;
    }
    if(!mm27_cache)return;
    const auto c=mm27_cache->counters();
    s.cache_hits=c.hits;s.cache_misses=c.misses;s.cache_evictions=c.evictions;s.cache_reuses=c.reuses;
    s.cache_oom=c.oom;s.cache_bypasses=c.bypasses+c.rejected;
    s.cache_pressure_trims=c.pressure_trims;s.cache_pressure_groups=c.pressure_groups;
    s.cache_pressure_evicted_bytes=c.pressure_evicted_bytes;s.cache_pressure_released_bytes=c.pressure_released_bytes;
    s.cache_resident=mm27_cache->resident_bytes();s.cache_limit=mm27_cache->budget();
    const auto g=mm27_cache->group_stats();s.cache_group_experts=mm27_cache->grouped();
    s.cache_expert_groups=g.groups;s.cache_partial_expert_groups=g.partial;
    s.cache_ready_matrices=g.ready_matrices;s.cache_pending_matrices=g.pending_matrices;
    s.cache_ready_bytes=g.ready_bytes;s.cache_group_admissions=g.admissions;
    if(mm27_arena) {
        const auto a=mm27_arena->snapshot();
        s.arena_reserved=a.reserved;s.arena_live=a.live;s.arena_blocks=a.blocks;
        s.arena_allocations=a.allocations;s.arena_frees=a.frees;s.arena_rejects=a.budget_rejects;
    }
}
