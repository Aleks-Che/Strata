#pragma once
#include "sync_runtime.h"
#include "contract.hpp"
#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hy3 {
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
inline void environment() {
    // Resident vs offloaded F32 batch4 differs with candidate CUDA fusions.
    // Keep a reproducible unfused P1 baseline; re-admit fusions separately.
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0"); _putenv_s("GGML_OP_OFFLOAD_MIN_BATCH", "1");
    _putenv_s("GGML_CUDA_DISABLE_FUSION", "1");
    // HY3-09: graph-enabled history replays can diverge despite identical
    // weights. Disable capture/replay until its exact failure is isolated.
    _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1); setenv("GGML_OP_OFFLOAD_MIN_BATCH", "1", 1);
    setenv("GGML_CUDA_DISABLE_FUSION", "1", 1);
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
#endif
}
inline bool expert(const std::string & name) {
    const std::string suffix = "_exps.weight";
    return name.size() >= suffix.size() && name.compare(name.size()-suffix.size(), suffix.size(), suffix) == 0;
}
inline Model load(const std::string & path, bool resident = false, bool cpu_embedding = false, bool fixture = false, bool mtp = false, bool stream_mtp = false) {
    const auto contract=inspect(path,fixture);
    strata_hy3_memory_check(contract.fixed_bytes+(resident ? contract.routed_bytes : 0)+(mtp ? contract.mtp_bytes-(stream_mtp ? contract.mtp_routed_bytes : 0) : 0));
    auto * gpu = ggml_backend_dev_by_name("CUDA0"), * cpu = ggml_backend_dev_by_name("CPU");
    require(gpu && cpu, "CUDA0 and CPU storage backends required");
    ggml_backend_dev_t devices[] = {gpu, nullptr};
    const std::string mtp_pattern="^blk\\."+std::to_string(contract.blocks-1)+"\\.";
    llama_model_tensor_buft_override overrides[] = {
        {"^token_embd\\.weight$", ggml_backend_dev_buffer_type(cpu_embedding ? cpu : gpu)},
        {mtp_pattern.c_str(), ggml_backend_dev_buffer_type(gpu)},
        {"^blk\\.[0-9]+\\.ffn_(gate|up|down)_exps\\.weight$", ggml_backend_dev_buffer_type(cpu)},
        {nullptr, nullptr}};
    // First matching override wins. Draft dense weights remain on GPU; only
    // its routed matrices join the same host source and cache as the target.
    if (stream_mtp) {
        overrides[1]=overrides[2];
        if(resident) {
            static thread_local std::string draft_experts;
            draft_experts=mtp_pattern+"ffn_(gate|up|down)_exps\\.weight$";
            overrides[1]={draft_experts.c_str(),ggml_backend_dev_buffer_type(cpu)};
        }
        overrides[2]={nullptr,nullptr};
    } else if (resident) overrides[2] = {nullptr, nullptr};
    auto mp = llama_model_default_params();
    mp.devices = devices; mp.tensor_buft_overrides = overrides; mp.n_gpu_layers = -1;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE; mp.load_mtp = mtp; mp.use_extra_bufts = false;
    mp.no_host = true; mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    Model model(llama_model_load_from_file(path.c_str(), mp), llama_model_free);
    require(bool(model), "Hy3 model load failed");
    require(model->arch == LLM_ARCH_HY_V3, "unexpected model architecture after load");
    size_t expert_count = 0;
    for (const auto & entry : model->tensors_by_name) {
        const auto * tensor = entry.second;
        require(tensor->data && tensor->buffer, "unallocated model weight: " + entry.first);
        const bool host = ggml_backend_buffer_is_host(tensor->buffer);
        if (expert(entry.first)) {
            ++expert_count;
            const bool draft=entry.first.rfind("blk."+std::to_string(contract.blocks-1)+".",0)==0;
            require(host != (draft ? !stream_mtp : resident), "wrong expert placement: " + entry.first);
        } else if (!(cpu_embedding && entry.first == "token_embd.weight")) {
            require(!host, "non-routed weight remained on CPU: " + entry.first);
        }
    }
    size_t routed_layers = 0;
    // Hy3 chooses dense/MoE by tensor presence; appended MTP is skipped.
    for (uint32_t i = 0; i < model->hparams.n_layer(); ++i) {
        const auto & layer = model->layers[i];
        if (layer.ffn_gate_exps) {
            require(layer.ffn_up_exps && layer.ffn_down_exps && layer.ffn_gate_inp, "incomplete Hy3 routed layer");
            ++routed_layers;
        } else require(layer.ffn_gate && layer.ffn_up && layer.ffn_down, "incomplete Hy3 dense layer");
    }
    require(routed_layers && expert_count == 3*(routed_layers+(mtp ? 1 : 0)), "unexpected Hy3 expert tensors");
    require(routed_layers == contract.blocks-2, "unexpected Hy3 main/MTP boundary");
    return model;
}
inline Context context(llama_model * model, int size = 2048, int batch = 17, ggml_type kv = GGML_TYPE_F32,
                       ggml_backend_sched_eval_callback callback=nullptr, void * callback_data=nullptr, bool draft=false) {
    require(size >= 32 && size <= 2048 && batch >= 1 && batch <= 32 && batch <= size, "invalid Hy3 context or batch");
    auto cp = llama_context_default_params();
    cp.n_ctx = size; cp.n_batch = cp.n_ubatch = batch; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 4; cp.type_k = cp.type_v = kv;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; cp.offload_kqv = cp.op_offload = true;
    cp.swa_full = false;
    cp.ctx_type=draft ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
    cp.cb_eval=callback;cp.cb_eval_user_data=callback_data;
    Context ctx(llama_init_from_model(model, cp), llama_free);
    require(bool(ctx), "Hy3 context creation failed");
    return ctx;
}
inline void configure_cache(llama_model * model,size_t cap) {
    strata_hy3_cache_begin(cap);
    if(cap) for(const auto & entry:model->tensors_by_name)
        if(expert(entry.first) && ggml_backend_buffer_is_host(entry.second->buffer)) strata_hy3_cache_register(entry.second,entry.first.rfind("blk."+std::to_string(model->hparams.n_layer())+".",0)==0);
}
inline void decode(llama_context * ctx, const std::vector<llama_token> & tokens,
                   size_t begin, int count, int position, bool all_logits = false) {
    require(count > 0 && begin <= tokens.size() && size_t(count) <= tokens.size()-begin, "invalid Hy3 decode slice");
    require(uint32_t(count)<=llama_n_batch(ctx),"decode slice exceeds context batch");
    strata_hy3_memory_check();
    auto batch = llama_batch_init(count, 0, 1); batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = tokens[begin+i]; batch.pos[i] = position+i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = all_logits || i == count-1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch); llama_synchronize(ctx);
    require(rc == 0, "Hy3 decode failed: " + std::to_string(rc));
}
inline void clear(llama_context * ctx) {
    llama_synchronize(ctx); llama_memory_clear(llama_get_memory(ctx), true);
}
} // namespace hy3
