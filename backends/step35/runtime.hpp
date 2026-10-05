#pragma once
#include "sync_runtime.h"
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

namespace step35 {
class RequestPhase {
    int previous;
public:
    explicit RequestPhase(int phase):previous(strata_step_request_phase(phase)) {}
    ~RequestPhase() {strata_step_request_phase(previous);}
    RequestPhase(const RequestPhase &)=delete;
    RequestPhase & operator=(const RequestPhase &)=delete;
    void decode() {strata_step_request_phase(2);}
};
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
inline void require(bool ok, const std::string & message) { if (!ok) throw std::runtime_error(message); }
inline void environment() {
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0"); _putenv_s("GGML_OP_OFFLOAD_MIN_BATCH", "1");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1); setenv("GGML_OP_OFFLOAD_MIN_BATCH", "1", 1);
#endif
}
inline bool expert(const std::string & name) {
    const std::string suffix = "_exps.weight";
    return name.size() >= suffix.size() && name.compare(name.size()-suffix.size(), suffix.size(), suffix) == 0;
}
inline Model load(const std::string & path, bool resident = false, bool cpu_embedding = false) {
    // Check architecture before allocating weights, including a metadata-only
    // first shard. The native loader resolves and owns the complete split set.
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_file(path.c_str(), {true, nullptr}), gguf_free);
    require(bool(metadata), "cannot read Step GGUF header");
    const auto arch = gguf_find_key(metadata.get(), "general.architecture");
    require(arch >= 0 && gguf_get_kv_type(metadata.get(), arch) == GGUF_TYPE_STRING &&
        std::strcmp(gguf_get_val_str(metadata.get(), arch), "step35") == 0, "Step engine requires architecture step35");
    metadata.reset();
    auto * gpu = ggml_backend_dev_by_name("CUDA0"), * cpu = ggml_backend_dev_by_name("CPU");
    require(gpu && cpu, "CUDA0 and CPU storage backends required");
    ggml_backend_dev_t devices[] = {gpu, nullptr};
    llama_model_tensor_buft_override overrides[] = {
        {"^token_embd\\.weight$", ggml_backend_dev_buffer_type(cpu_embedding ? cpu : gpu)},
        {"^blk\\.[0-9]+\\.ffn_(gate|up|down)_exps\\.weight$", ggml_backend_dev_buffer_type(cpu)},
        {nullptr, nullptr}};
    if (resident) overrides[1] = {nullptr, nullptr};
    auto mp = llama_model_default_params();
    mp.devices = devices; mp.tensor_buft_overrides = overrides; mp.n_gpu_layers = -1;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE; mp.load_mtp = false; mp.use_extra_bufts = false;
    mp.no_host = true; mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    Model model(llama_model_load_from_file(path.c_str(), mp), llama_model_free);
    require(bool(model), "Step model load failed");
    require(model->arch == LLM_ARCH_STEP35, "unexpected model architecture after load");
    size_t expert_count = 0;
    for (const auto & entry : model->tensors_by_name) {
        const auto * tensor = entry.second;
        require(tensor->data && tensor->buffer, "unallocated model weight: " + entry.first);
        const bool host = ggml_backend_buffer_is_host(tensor->buffer);
        if (expert(entry.first)) {
            ++expert_count;
            require(host != resident, "wrong expert placement: " + entry.first);
        } else if (!(cpu_embedding && entry.first == "token_embd.weight")) {
            require(!host, "non-routed weight remained on CPU: " + entry.first);
        }
    }
    size_t routed_layers = 0;
    // Step chooses dense/MoE by tensor presence; this pin does not populate
    // n_layer_dense_lead for Step (the similarly named GLM field is not valid).
    for (uint32_t i = 0; i < model->hparams.n_layer(); ++i) {
        const auto & layer = model->layers[i];
        if (layer.ffn_gate_exps) {
            require(layer.ffn_up_exps && layer.ffn_down_exps && layer.ffn_gate_inp, "incomplete Step routed layer");
            ++routed_layers;
        } else require(layer.ffn_gate && layer.ffn_up && layer.ffn_down, "incomplete Step dense layer");
    }
    require(routed_layers && expert_count == 3*routed_layers, "unexpected Step expert tensors");
    return model;
}
inline Context context(llama_model * model, int size = 2048, int batch = 16, ggml_type kv = GGML_TYPE_F32) {
    require(size >= 32 && size <= 32768 && batch >= 1 && batch <= 256 && batch <= size, "invalid Step context or batch");
    auto cp = llama_context_default_params();
    cp.n_ctx = size; cp.n_batch = cp.n_ubatch = batch; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 4; cp.type_k = cp.type_v = kv;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; cp.offload_kqv = cp.op_offload = true;
    cp.swa_full = false;
    Context ctx(llama_init_from_model(model, cp), llama_free);
    require(bool(ctx), "Step context creation failed");
    return ctx;
}
inline void decode(llama_context * ctx, const std::vector<llama_token> & tokens,
                   size_t begin, int count, int position, bool all_logits = false) {
    require(count > 0 && begin <= tokens.size() && size_t(count) <= tokens.size()-begin, "invalid Step decode slice");
    strata_step_cache_refresh();
    auto batch = llama_batch_init(count, 0, 1); batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = tokens[begin+i]; batch.pos[i] = position+i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = all_logits || i == count-1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch); llama_synchronize(ctx);
    require(rc == 0, "Step decode failed: " + std::to_string(rc));
}
inline void clear(llama_context * ctx) {
    llama_synchronize(ctx); llama_memory_clear(llama_get_memory(ctx), true);
}
} // namespace step35
