// Included only by Hy3's private scheduler. Reuse Step's tested allocation,
// frequency and budget controller without changing the Step implementation.
#include "../step35/expert_cache.hpp"
#include <array>
struct Hy3TensorIdentity {
    uint32_t id;
    std::string name;
    ggml_type type;
    std::array<int64_t,4> shape;
    std::array<size_t,4> strides;
    std::shared_ptr<strata_expert_file::Source> file;
    uint64_t offset;
};
static thread_local std::unique_ptr<step35::ExpertCache> hy3_cache;
static thread_local std::map<const void *,Hy3TensorIdentity> hy3_tensors;
static thread_local uint64_t hy3_generation=0;
static thread_local cudaStream_t hy3_cache_stream=nullptr;
static thread_local bool hy3_profile_delivery=false;
static thread_local bool hy3_prefill=false,hy3_admit_prefill=true;
static bool hy3_expert_weight(const ggml_tensor *);
static bool hy3_cancelled();
static bool hy3_copy_uncached(ggml_backend_t,ggml_tensor *,const ggml_tensor *,size_t,size_t);
static bool hy3_pipeline_enabled();
static bool hy3_pipeline_copy(ggml_backend_t,ggml_tensor *,const ggml_tensor *,size_t,size_t);
static void hy3_account_pipeline();
void strata_hy3_gpu_cache_policy(bool admit_prefill) {
    if(hy3_pipeline_enabled())throw std::runtime_error("configure Hy3 GPU cache policy before readers");
    hy3_admit_prefill=admit_prefill;
}
static void *hy3_cache_get(const step35::MatrixKey &key,size_t bytes) {
    return hy3_cache->get(key,bytes,hy3_admit_prefill || !hy3_prefill);
}
static void *hy3_cache_admit(const step35::MatrixKey &key,size_t bytes) {
    if(hy3_prefill && !hy3_admit_prefill) {++hy3_sync_stats.cache_prefill_bypasses;return nullptr;}
    return hy3_cache->admit(key,bytes);
}

void strata_hy3_delivery_profile(bool enabled) {
    hy3_profile_delivery=enabled;
    if(hy3_cache) hy3_cache->set_profile(enabled);
}

void strata_hy3_cache_begin(size_t cap) {
    strata_hy3_pipeline_config(0);
    hy3_host_cache.reset();
    if(hy3_cache_stream) {
        hy3_cuda(cudaStreamSynchronize(hy3_cache_stream));
        hy3_cuda(cudaStreamDestroy(hy3_cache_stream));hy3_cache_stream=nullptr;
    }
    hy3_cache.reset();hy3_tensors.clear();
    if(++hy3_generation==0) throw std::runtime_error("Hy3 cache generation overflow");
    if(!cap) return;
    if(hy3_sync_mode!=2) throw std::runtime_error("Hy3 cache requires pinned copy mode");
    hy3_cache=std::make_unique<step35::ExpertCache>(cap,[] {
        auto s=step35::memory_sample();
        s.gpu_free=std::min(s.gpu_free,hy3_test_limits.gpu_available);
        s.ram_free=std::min(s.ram_free,hy3_test_limits.ram_available);
        return s;
    },[](void ** p,size_t n) {
        if(hy3_test_limits.fail_next_cache_allocation) {
            hy3_test_limits.fail_next_cache_allocation=false;return cudaErrorMemoryAllocation;
        }
        return cudaMalloc(p,n);
    });
    hy3_cache->set_reuse_allocations(true);
    hy3_cache->set_fast_scan(true);
    hy3_cache->set_profile(hy3_profile_delivery);
    hy3_cuda(cudaStreamCreateWithFlags(&hy3_cache_stream,cudaStreamNonBlocking));
    hy3_cache->refresh();
}
void strata_hy3_cache_register(const ggml_tensor * t) {
    if(!hy3_cache || !t || !t->data || !t->buffer || !ggml_backend_buffer_is_host(t->buffer) ||
        !hy3_expert_weight(t) || !ggml_is_contiguous(t) || t->ne[3]!=1 || t->ne[2]<=0 || !t->nb[2])
        throw std::runtime_error("invalid Hy3 cache tensor registration");
    auto file=strata_expert_file::find(t->data,ggml_nbytes(t));
    if(!file) throw std::runtime_error("Hy3 cache tensor has no live source mapping");
    const auto offset=file->offset_of(t->data,ggml_nbytes(t));
    Hy3TensorIdentity id{uint32_t(hy3_tensors.size()),t->name,t->type,{}, {},file,offset};
    std::copy(t->ne,t->ne+4,id.shape.begin());std::copy(t->nb,t->nb+4,id.strides.begin());
    if(!hy3_tensors.emplace(t->data,std::move(id)).second)
        throw std::runtime_error("duplicate Hy3 cache tensor registration");
}
static bool hy3_copy_range(ggml_backend_t backend,ggml_tensor * dst,const ggml_tensor * src,size_t offset,size_t bytes) {
    if(hy3_pipeline_enabled()) return hy3_pipeline_copy(backend,dst,src,offset,bytes);
    if(!hy3_cache || !hy3_cache->budget()) return hy3_copy_uncached(backend,dst,src,offset,bytes);
    try {
        if(hy3_cancelled()) return false;
        const auto entry=hy3_tensors.find(src->data);
        if(entry==hy3_tensors.end()) throw std::runtime_error("unregistered Hy3 cache tensor");
        const auto & t=entry->second;
        const auto file=strata_expert_file::find(src->data,ggml_nbytes(src));
        if(!file || file!=t.file || file->offset_of(src->data,ggml_nbytes(src))!=t.offset ||
            t.name!=src->name || t.type!=src->type || dst->type!=src->type ||
            !std::equal(t.shape.begin(),t.shape.end(),src->ne) || !std::equal(t.strides.begin(),t.strides.end(),src->nb) ||
            !ggml_are_same_shape(src,dst) || !ggml_is_contiguous(dst) || ggml_backend_buffer_is_host(dst->buffer) ||
            offset%t.strides[2] || offset>ggml_nbytes(src) || bytes>ggml_nbytes(src)-offset)
            throw std::runtime_error("Hy3 cache tensor identity/range changed");
        const size_t stride=t.strides[2],count=bytes/stride,tail=bytes%stride;
        if(!count || (tail && tail!=std::min<size_t>(512,stride)))
            throw std::runtime_error("invalid Hy3 cache range padding");
        ggml_backend_synchronize(backend); // Complete scratch consumers first.
        for(size_t i=0;i<count;++i) {
            if(hy3_cancelled()) return false;
            const size_t expert=offset/stride+i,at=expert*stride;
            // Store native MMQ's actual next-expert padding, never invented zeros.
            const size_t stored=stride+(expert+1<size_t(t.shape[2]) ? std::min<size_t>(512,stride) : 0);
            const step35::MatrixKey key{hy3_generation,t.id,uint32_t(expert)};
            if(auto * hit=hy3_cache_get(key,stored)) {
                hy3_cuda(cudaMemcpyAsync(static_cast<uint8_t *>(dst->data)+at,hit,stored,cudaMemcpyDeviceToDevice,hy3_cache_stream));
                hy3_cuda(cudaStreamSynchronize(hy3_cache_stream));
                hy3_sync_stats.d2d_bytes+=stored;++hy3_sync_stats.ranges;
                if(hy3_observer) hy3_observer(hy3_owner,backend,dst,src,at,stored);
            } else {
                if(!hy3_copy_uncached(backend,dst,src,at,stored)) return false;
                // Admission follows the complete upload: partial reads never publish entries.
                if(auto * slot=hy3_cache_admit(key,stored)) {
                    if(hy3_test_limits.fail_next_cache_fill) {
                        hy3_test_limits.fail_next_cache_fill=false;
                        throw std::runtime_error("injected Hy3 cache fill failure");
                    }
                    hy3_cuda(cudaMemcpyAsync(slot,static_cast<uint8_t *>(dst->data)+at,stored,cudaMemcpyDeviceToDevice,hy3_cache_stream));
                    hy3_cuda(cudaStreamSynchronize(hy3_cache_stream));
                    hy3_sync_stats.cache_fill_bytes+=stored;
                }
            }
        }
        return true;
    } catch(const std::exception & error) {
        // Fence before discarding any partially filled slot or cached source.
        if(hy3_cache_stream) cudaStreamSynchronize(hy3_cache_stream);
        if(hy3_cache) hy3_cache->trim(0);
        GGML_LOG_ERROR("STRATA_HY3 cache transfer: %s\n",error.what());return false;
    }
}
