#pragma once
// Small, deterministic F32 model for executing the candidate's real hybrid graph.
// These are synthetic weights; this fixture says nothing about model quality.
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

inline void write_synthetic_glm(const std::string & path, uint32_t top_k) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ggml_tensor_overhead()*128, nullptr, true}), ggml_free);
    if (!file || !ctx) throw std::runtime_error("fixture metadata allocation failed");
    auto * f = file.get();
    gguf_set_val_str(f, "general.architecture", "glm5next");
    gguf_set_val_str(f, "general.name", "Strata synthetic hybrid validation fixture");
    gguf_set_val_str(f, "tokenizer.ggml.model", "none");
    const auto u = [f](const char * key, uint32_t value) {
        gguf_set_val_u32(f, (std::string("glm5next.")+key).c_str(), value);
    };
    const auto v = [f](const char * key, float value) {
        gguf_set_val_f32(f, (std::string("glm5next.")+key).c_str(), value);
    };
    u("block_count", 2); u("context_length", 512); u("embedding_length", 256);
    u("feed_forward_length", 512); u("vocab_size", 64); u("attention.head_count", 2);
    const uint32_t kv_heads[] = {0, 1};
    gguf_set_arr_data(f, "glm5next.attention.head_count_kv", GGUF_TYPE_UINT32, kv_heads, 2);
    u("attention.q_lora_rank", 128); u("attention.kv_lora_rank", 128);
    u("attention.key_length", 128); u("attention.value_length", 128);
    u("attention.key_length_mla", 128); u("attention.value_length_mla", 128);
    u("rope.dimension_count", 0); u("ssm.conv_kernel", 4); u("kda.head_dim", 128);
    v("kda.gate_lower_bound", -5); v("attention.layer_norm_rms_epsilon", 1e-5f);
    v("attention.layer_norm_epsilon", 1e-6f);
    // Preserve the real indexer geometry: CUDA fused LID supports 32/64 heads.
    u("attention.indexer.head_count", 32); u("attention.indexer.key_length", 128);
    u("attention.indexer.top_k", top_k); u("attention.indexer.kpool", 4);
    u("hyper_connection.count", 4); u("hyper_connection.sinkhorn_iterations", 20);
    v("hyper_connection.epsilon", 1e-6f);
    u("expert_count", 4); u("expert_used_count", 2); u("expert_shared_count", 1);
    u("expert_feed_forward_length", 256); u("expert_shared_feed_forward_length", 256);
    u("expert_group_count", 1); u("expert_group_used_count", 1); u("expert_gating_func", 2);
    u("leading_dense_block_count", 1); u("nextn_predict_layers", 0);
    v("expert_weights_scale", 2.5f);
    gguf_set_val_bool(f, "glm5next.expert_weights_norm", true);
    v("swiglu_clamp_exp", 10); v("swiglu_clamp_shexp", 10);

    uint32_t rng = 0x53c001u;
    std::vector<std::vector<float>> weights;
    const auto add = [&](const std::string & name, int64_t x, int64_t y=1, int64_t z=1) {
        auto * t = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, x, y, z);
        ggml_set_name(t, name.c_str());
        weights.emplace_back(ggml_nelements(t));
        const bool norm = name.find("norm.weight") != std::string::npos;
        const float scale = y == 1 && z == 1 ? 0.1f : 0.35f/std::sqrt(float(x));
        for (auto & value : weights.back()) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            value = (float(rng & 0xffffu)/32767.5f - 1.0f)*scale;
            if (norm) value += 1.0f;
            if (name == "blk.0.ssm_a") value = -0.8f + value;
            if (name.find("hc_") != std::string::npos && name.find("_scale") != std::string::npos)
                value += 0.7f;
        }
        t->data = weights.back().data();
        gguf_add_tensor(f, t);
    };
    add("token_embd.weight", 256, 64); add("output.weight", 256, 64); add("output_norm.weight", 256);
    for (int il=0; il<2; ++il) {
        const std::string p = "blk."+std::to_string(il)+".";
        add(p+"attn_norm.weight", 256); add(p+"ffn_norm.weight", 256);
        for (const std::string kind : {"attn", "ffn"}) {
            add(p+"hc_"+kind+"_fn.weight", 1024, 24);
            add(p+"hc_"+kind+"_base.weight", 24); add(p+"hc_"+kind+"_scale.weight", 3);
        }
        add(p+"attn_output.weight", 256, 256);
    }
    for (const std::string qkv : {"q", "k", "v"}) {
        add("blk.0.attn_"+qkv+".weight", 256, 256);
        add("blk.0.ssm_conv1d_"+qkv+".weight", 4, 1, 256);
    }
    for (const std::string fg : {"f", "g"}) {
        add("blk.0.ssm_"+fg+"_a.weight", 256, 128);
        add("blk.0.ssm_"+fg+"_b.weight", 128, 256);
    }
    add("blk.0.ssm_beta.weight", 256, 2); add("blk.0.ssm_a", 2);
    add("blk.0.ssm_dt.bias", 256); add("blk.0.ssm_norm.weight", 128);
    add("blk.0.ffn_gate.weight", 256, 512); add("blk.0.ffn_up.weight", 256, 512);
    add("blk.0.ffn_down.weight", 512, 256);
    add("blk.1.attn_q_a.weight", 256, 128); add("blk.1.attn_q_a_norm.weight", 128);
    add("blk.1.attn_q_b.weight", 128, 256); add("blk.1.attn_kv_a_mqa.weight", 256, 128);
    add("blk.1.attn_kv_a_norm.weight", 128); add("blk.1.attn_k_b.weight", 128, 128, 2);
    add("blk.1.attn_v_b.weight", 128, 128, 2);
    add("blk.1.indexer.k_norm.weight", 128); add("blk.1.indexer.k_norm.bias", 128);
    add("blk.1.indexer.proj.weight", 256, 32); add("blk.1.indexer.attn_k.weight", 256, 128);
    add("blk.1.indexer.attn_q_b.weight", 128, 4096);
    add("blk.1.indexer_compressor_gate.weight", 256, 128);
    add("blk.1.indexer_compressor_ape.weight", 128, 4);
    add("blk.1.ffn_gate_inp.weight", 256, 4); add("blk.1.exp_probs_b.bias", 4);
    for (const std::string kind : {"gate", "up", "down"}) {
        add("blk.1.ffn_"+kind+"_exps.weight", 256, 256, 4);
        add("blk.1.ffn_"+kind+"_shexp.weight", 256, 256);
    }
    if (!gguf_write_to_file(f, path.c_str(), false)) throw std::runtime_error("cannot write synthetic GGUF");
}
