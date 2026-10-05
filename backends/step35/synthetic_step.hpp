#pragma once
// Deterministic Step decoder fixture: real loader/graph, synthetic F32 weights.
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

inline void write_synthetic_step(const std::string & path, bool mixed_quant = false, bool wide_experts = false) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ggml_tensor_overhead() * 128, nullptr, true}), ggml_free);
    if (!file || !ctx) throw std::runtime_error("Step fixture metadata allocation failed");
    auto * f = file.get();
    gguf_set_val_str(f, "general.architecture", "step35");
    gguf_set_val_str(f, "general.name", "Strata synthetic Step full-SWA graph fixture");
    gguf_set_val_str(f, "tokenizer.ggml.model", "none");
    auto u = [f](const char * key, uint32_t value) { gguf_set_val_u32(f, (std::string("step35.") + key).c_str(), value); };
    auto v = [f](const char * key, float value) { gguf_set_val_f32(f, (std::string("step35.") + key).c_str(), value); };
    u("block_count", 3); u("context_length", 2048); u("embedding_length", 256);
    u("feed_forward_length", 512); u("vocab_size", 64);
    const uint32_t heads[] = {2, 3, 2}, kv[] = {1, 1, 1};
    const bool swa[] = {false, true, false};
    const float routed_clamp[] = {0, 0, 7}, shared_clamp[] = {0, 0, 16};
    gguf_set_arr_data(f, "step35.attention.head_count", GGUF_TYPE_UINT32, heads, 3);
    gguf_set_arr_data(f, "step35.attention.head_count_kv", GGUF_TYPE_UINT32, kv, 3);
    gguf_set_arr_data(f, "step35.attention.sliding_window_pattern", GGUF_TYPE_BOOL, swa, 3);
    gguf_set_arr_data(f, "step35.swiglu_clamp_exp", GGUF_TYPE_FLOAT32, routed_clamp, 3);
    gguf_set_arr_data(f, "step35.swiglu_clamp_shexp", GGUF_TYPE_FLOAT32, shared_clamp, 3);
    u("attention.key_length", 128); u("attention.value_length", 128);
    u("attention.sliding_window", 512); v("attention.layer_norm_rms_epsilon", 1e-5f);
    v("rope.freq_base", 5000000); v("rope.freq_base_swa", 10000);
    u("expert_count", 16); u("expert_used_count", 8); u("leading_dense_block_count", 1);
    const int expert_ff = wide_experts ? 2304 : 256;
    u("expert_feed_forward_length", expert_ff); u("expert_shared_feed_forward_length", 256);
    u("expert_gating_func", 2); u("moe_every_n_layers", 1); v("expert_weights_scale", 3);
    gguf_set_val_bool(f, "step35.expert_weights_norm", true);

    uint32_t rng = 0x37d001u;
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<uint8_t>> packed;
    const auto add = [&](const std::string & name, int64_t x, int64_t y = 1, int64_t z = 1) {
        const auto type = !mixed_quant || y == 1 ? GGML_TYPE_F32 : z > 1 ? GGML_TYPE_Q4_K :
            name == "output.weight" ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0;
        auto * t = ggml_new_tensor_3d(ctx.get(), type, x, y, z);
        ggml_set_name(t, name.c_str());
        weights.emplace_back(ggml_nelements(t));
        const bool norm = name.find("norm.weight") != std::string::npos;
        const bool strong = name.find("blk.2.ffn_up_") == 0 || name.find("blk.2.ffn_gate_") == 0;
        const bool router = name.find("ffn_gate_inp") != std::string::npos;
        const float scale = y == 1 && z == 1 ? 0.1f : (strong && !router ? 30.0f : 0.35f) / std::sqrt(float(x));
        size_t index = 0;
        for (auto & value : weights.back()) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            value = (float(rng & 0xffffu) / 32767.5f - 1) * scale;
            if (norm) value += 1;
            if (name == "rope_freqs.weight") value = 1 + float(index) * 0.007f;
            // Deliberately separated router scores avoid accidental near-ties
            // obscuring numerical/state tests. P0.4 separately tests varied IDs.
            if (name.find("exp_probs_b.bias") != std::string::npos) value = 0.2f * float(index) - 1.5f;
            ++index;
        }
        if (type == GGML_TYPE_F32) t->data = weights.back().data();
        else {
            packed.emplace_back(ggml_nbytes(t));
            std::vector<float> importance(x, 1.0f);
            if (ggml_quantize_chunk(type, weights.back().data(), packed.back().data(), 0, y*z, x, importance.data()) != packed.back().size())
                throw std::runtime_error("Step fixture quantization failed");
            t->data = packed.back().data();
        }
        gguf_add_tensor(f, t);
    };
    add("token_embd.weight", 256, 64); add("output.weight", 256, 64);
    add("output_norm.weight", 256); add("rope_freqs.weight", 64);
    for (int layer = 0; layer < 3; ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        add(p + "attn_norm.weight", 256); add(p + "ffn_norm.weight", 256);
        add(p + "attn_q_norm.weight", 128); add(p + "attn_k_norm.weight", 128);
        add(p + "attn_q.weight", 256, heads[layer] * 128);
        add(p + "attn_k.weight", 256, 128); add(p + "attn_v.weight", 256, 128);
        add(p + "attn_output.weight", heads[layer] * 128, 256);
        add(p + "attn_gate.weight", 256, heads[layer]);
        if (layer == 0) {
            add(p + "ffn_gate.weight", 256, 512); add(p + "ffn_up.weight", 256, 512);
            add(p + "ffn_down.weight", 512, 256);
        } else {
            add(p + "ffn_gate_inp.weight", 256, 16); add(p + "exp_probs_b.bias", 16);
            for (const std::string kind : {"gate", "up", "down"}) {
                add(p + "ffn_" + kind + "_exps.weight", kind == "down" ? expert_ff : 256, kind == "down" ? 256 : expert_ff, 16);
                add(p + "ffn_" + kind + "_shexp.weight", 256, 256);
            }
        }
    }
    if (!gguf_write_to_file(f, path.c_str(), false)) throw std::runtime_error("cannot write Step fixture");
}
