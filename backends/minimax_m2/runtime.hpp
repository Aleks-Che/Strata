#pragma once
#include "contract.hpp"
#include "sync_runtime.h"
#include "llama.h"
#include "llama-model.h"
#include <cstdlib>
namespace minimax_m2 {
using Model=std::unique_ptr<llama_model,decltype(&llama_model_free)>;
using Context=std::unique_ptr<llama_context,decltype(&llama_free)>;
inline void free_model(llama_model *model) {strata_mm27_cache_unbind(model);llama_model_free(model);}
// Export EOS/PAD; native FIM/reponame EOG aliases are not conversation stops.
inline bool is_stop(llama_token token) {return token==200020;}
inline void environment() {
    // In-process _putenv_s was insufficient for CUDA's strict numerical path on
    // Windows. Require inherited values, set by the parent before process start.
    for(const auto &p:std::vector<std::pair<const char *,const char *>>{
        {"NVIDIA_TF32_OVERRIDE","0"},{"GGML_CUDA_CUBLAS_COMPUTE_TYPE","f32"},
        {"GGML_CUDA_DISABLE_GRAPHS","1"},{"LLAMA_GRAPH_REUSE_DISABLE","1"},
        {"GGML_OP_OFFLOAD_MIN_BATCH","1"},{"STRATA_MM27_QUANT_F32","1"}}) {
        const char *value=std::getenv(p.first);
        require(value && std::string(value)==p.second,std::string("set ")+p.first+"="+p.second+
            " before process start (use tools/run_minimax_m2.py)");
    }
}
inline bool expert(const std::string &name) {return name.size()>=12 && name.compare(name.size()-12,12,"_exps.weight")==0;}
inline Model load(const std::string &path,bool resident=false,bool fixture=false) {
    const auto contract=inspect(path,fixture);
    // Initial reserve; live checks run again after allocations and every decode.
    // This is admission, not a claim that scratch has exactly this size.
    strata_mm27_memory(contract.fixed_bytes+(resident?contract.routed_bytes:0)+(fixture?512ull<<20:8ull<<30),256ull<<20);
    auto *gpu=ggml_backend_dev_by_name("CUDA0"),*cpu=ggml_backend_dev_by_name("CPU");
    require(gpu && cpu,"CUDA0 and CPU storage backends required");
    ggml_backend_dev_t devices[]={gpu,nullptr};
    llama_model_tensor_buft_override overrides[]={
        {"^token_embd\\.weight$",ggml_backend_dev_buffer_type(gpu)},
        {"^blk\\.[0-9]+\\.ffn_(gate|up|down)_exps\\.weight$",ggml_backend_dev_buffer_type(cpu)},
        {nullptr,nullptr}};
    if(resident)overrides[1]={nullptr,nullptr};
    auto mp=llama_model_default_params();mp.devices=devices;mp.tensor_buft_overrides=overrides;
    mp.n_gpu_layers=-1;mp.split_mode=LLAMA_SPLIT_MODE_NONE;mp.load_mtp=false;
    mp.use_extra_bufts=false;mp.no_host=true;mp.load_mode=LLAMA_LOAD_MODE_MMAP;
    Model model(llama_model_load_from_file(path.c_str(),mp),free_model);
    require(model && model->arch==LLM_ARCH_MINIMAX_M2,"MiniMax model load failed");
    size_t routed=0;std::vector<const ggml_tensor *> cache_tensors;
    for(const auto &entry:model->tensors_by_name) {
        const auto *t=entry.second;require(t->data && t->buffer,"unallocated tensor: "+entry.first);
        const bool host=ggml_backend_buffer_is_host(t->buffer);
        if(expert(entry.first)) {++routed;require(host!=resident,"wrong expert placement: "+entry.first);if(host)cache_tensors.push_back(t);}
        else require(!host,"non-routed weight remained on CPU: "+entry.first);
    }
    require(routed==3*contract.blocks,"unexpected routed tensor count");
    if(!resident)strata_mm27_cache_bind(model.get(),cache_tensors.data(),cache_tensors.size());
    strata_mm27_memory();return model;
}
inline Context context(llama_model *model,int size=512,int batch=8) {
    require(size>=256 && size<=4096 && batch>=1 && batch<=16 && batch<=size,"invalid MiniMax context/batch");
    strata_mm27_cache_clear(); // KV/workspace allocation takes priority over cache.
    auto cp=llama_context_default_params();cp.n_ctx=size;cp.n_batch=cp.n_ubatch=batch;cp.n_seq_max=1;
    cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F32;
    // FA casts F32 KV to F16 and changes its kernel with batch size. Keep the
    // strict baseline on the already-validated full-F32 attention path.
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=true;
    Context ctx(llama_init_from_model(model,cp),llama_free);require(bool(ctx),"MiniMax context allocation failed");
    // Fresh contexts must have the same initialized KV storage as fresh pipe
    // requests. Allocator-reused F32 KV pages are otherwise history-dependent.
    llama_synchronize(ctx.get());llama_memory_clear(llama_get_memory(ctx.get()),true);
    strata_mm27_memory();return ctx;
}
inline void clear(llama_context *ctx) {llama_synchronize(ctx);llama_memory_clear(llama_get_memory(ctx),true);}
inline void decode(llama_context *ctx,const std::vector<llama_token> &tokens,size_t begin,int count,int position,bool all=false) {
    require(count>0 && begin<=tokens.size() && size_t(count)<=tokens.size()-begin,"invalid decode slice");
    strata_mm27_memory();auto batch=llama_batch_init(count,0,1);batch.n_tokens=count;
    for(int i=0;i<count;++i) {
        batch.token[i]=tokens[begin+i];batch.pos[i]=position+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=all || i==count-1;
    }
    const int rc=llama_decode(ctx,batch);llama_batch_free(batch);llama_synchronize(ctx);
    require(rc==0,"MiniMax decode failed: "+std::to_string(rc)+" "+strata_mm27_last_error());
    strata_mm27_memory();
}
} // namespace minimax_m2
