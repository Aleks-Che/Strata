#pragma once
// Shared placement/precision/decode contract for the baseline and pipe engine.
#include "sync_runtime.h"
#include "llama.h"
#include "ggml-backend.h"
#include "expert_load.hpp"
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata_glm {
using Model = std::shared_ptr<llama_model>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
// Literal diagnostics in per-logit checks must not allocate on the success path.
inline void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
inline void require(bool ok, const std::string & message) { if (!ok) throw std::runtime_error(message); }
inline bool mmvq_token_batch_enabled() {
    const char * value=std::getenv("STRATA_GLM_MMVQ_TOKEN_BATCH");
    require(!value || std::string(value)=="0" || std::string(value)=="1","STRATA_GLM_MMVQ_TOKEN_BATCH must be 0 or 1");
    return value && std::string(value)=="1";
}
inline void environment() {
    (void)mmvq_token_batch_enabled();
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0"); _putenv_s("GGML_OP_OFFLOAD_MIN_BATCH", "1");
    _putenv_s("STRATA_GLM_TOKENWISE_MATMUL", "1");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1); setenv("GGML_OP_OFFLOAD_MIN_BATCH", "1", 1);
    setenv("STRATA_GLM_TOKENWISE_MATMUL", "1", 1);
#endif
}
inline Model load(const std::string & path, bool resident=false, bool cpu_embedding=false, bool mtp=false) {
    const auto ram_layers=ram_expert_layers();
    auto * gpu=ggml_backend_dev_by_name("CUDA0"); auto * cpu=ggml_backend_dev_by_name("CPU");
    require(gpu && cpu, "CUDA0 and CPU buffer backends required");
    auto mp=llama_model_default_params();
    ggml_backend_dev_t devices[]={gpu,nullptr}; mp.devices=devices;
    llama_model_tensor_buft_override overrides[]={
        {"token_embd\\.weight",ggml_backend_dev_buffer_type(cpu_embedding ? cpu : gpu)},
        {"blk\\.[0-9]+\\.ffn_(gate|up|down)_exps\\.weight",ggml_backend_dev_buffer_type(cpu)},
        {nullptr,nullptr}};
    if (resident) overrides[1]={nullptr,nullptr};
    mp.tensor_buft_overrides=overrides; mp.n_gpu_layers=-1; mp.split_mode=LLAMA_SPLIT_MODE_NONE;
    mp.load_mode=ram_experts()?LLAMA_LOAD_MODE_NONE:LLAMA_LOAD_MODE_MMAP;
    // CPU overrides otherwise prefer CUDA_Host: never pin the whole expert set.
    mp.no_host=ram_experts();
    mp.lazy_mode=ram_experts() && ram_layers>=0 && !resident?LLAMA_LAZY_MODE_ON:LLAMA_LAZY_MODE_OFF;
    mp.load_mtp=mtp; mp.use_extra_bufts=false;
    Model model(llama_model_load_from_file(path.c_str(),mp),llama_model_free);
    require(bool(model),"model load failed");
    char arch[64]{}; llama_model_meta_val_str(model.get(),"general.architecture",arch,sizeof(arch));
    require(std::string(arch)=="glm5next","this backend accepts glm5next only");
    return model;
}
inline Context context(llama_model * model, int size, int batch, int threads=4, int rollback=0, bool mtp=false,
                       ggml_backend_sched_eval_callback callback=nullptr,void * callback_data=nullptr) {
    auto cp=llama_context_default_params();
    cp.n_ctx=size; cp.n_batch=cp.n_ubatch=batch; cp.n_seq_max=1; cp.n_rs_seq=rollback;
    if (mtp) cp.ctx_type=LLAMA_CONTEXT_TYPE_MTP;
    cp.n_threads=cp.n_threads_batch=threads; cp.type_k=cp.type_v=GGML_TYPE_F16;
    cp.cb_eval=callback;cp.cb_eval_user_data=callback_data;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED; cp.offload_kqv=cp.op_offload=true; cp.no_perf=false;
    Context ctx(llama_init_from_model(model,cp),llama_free);
    require(bool(ctx),"context allocation failed");
    return ctx;
}
inline void decode(llama_context * ctx, const std::vector<llama_token> & ids, int start, int count, int position, bool all_logits=false) {
    auto b=llama_batch_init(count,0,1); b.n_tokens=count;
    for (int i=0;i<count;++i) {
        b.token[i]=ids[start+i]; b.pos[i]=position+i; b.n_seq_id[i]=1; b.seq_id[i][0]=0; b.logits[i]=all_logits || i==count-1;
    }
    const int rc=llama_decode(ctx,b); llama_batch_free(b);
    require(rc==0,"llama_decode failed: "+std::to_string(rc));
    llama_synchronize(ctx);
}
inline void clear(llama_context * ctx) {
    llama_synchronize(ctx);
    llama_memory_clear(llama_get_memory(ctx),true);
}
} // namespace strata_glm
