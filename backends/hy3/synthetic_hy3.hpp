#pragma once
// Tiny deterministic model for the real hy_v3 loader, main graph and MTP graph.
// Dimensions are reduced; attention, routing and NextN conventions match Hy3.
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

inline void write_synthetic_hy3(const std::string & path, bool mixed) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ggml_tensor_overhead()*128, nullptr, true}), ggml_free);
    if (!file || !ctx) throw std::runtime_error("Hy3 fixture allocation failed");
    auto * f = file.get();
    gguf_set_val_str(f, "general.architecture", "hy_v3");
    gguf_set_val_str(f, "general.name", "Strata synthetic Hy3 2+1 full-GQA NextN fixture");
    gguf_set_val_str(f, "tokenizer.ggml.model", "none");
    auto u = [f](const char * key, uint32_t x) { gguf_set_val_u32(f, (std::string("hy_v3.")+key).c_str(), x); };
    auto v = [f](const char * key, float x) { gguf_set_val_f32(f, (std::string("hy_v3.")+key).c_str(), x); };
    u("block_count", 3); u("nextn_predict_layers", 1); u("context_length", 2048);
    u("embedding_length", 256); u("feed_forward_length", 512); u("vocab_size", 64);
    u("attention.head_count", 4); u("attention.head_count_kv", 1);
    u("attention.key_length", 128); u("attention.value_length", 128);
    u("rope.dimension_count", 128); v("rope.freq_base", 11158840);
    v("attention.layer_norm_rms_epsilon", 1e-5f);
    u("expert_count", 16); u("expert_used_count", 8);
    u("expert_feed_forward_length", 256); u("expert_shared_feed_forward_length", 256);
    u("expert_gating_func", 2); v("expert_weights_scale", 2.826f);
    gguf_set_val_bool(f, "hy_v3.expert_weights_norm", true);
    uint32_t rng = 0x483312u;
    std::vector<std::vector<float>> values;
    std::vector<std::vector<uint8_t>> packed;
    const auto add = [&](const std::string & name, int64_t x, int64_t y = 1, int64_t z = 1) {
        ggml_type type = GGML_TYPE_F32;
        const bool router = name.find("ffn_gate_inp") != std::string::npos;
        if (mixed && y > 1 && !router) {
            type = GGML_TYPE_Q8_0;
            if (name == "token_embd.weight" || name == "output.weight") type = GGML_TYPE_Q6_K;
            if (z > 1) {
                const bool down = name.find("ffn_down") != std::string::npos;
                const bool mtp = name.rfind("blk.2.", 0) == 0;
                type = mtp ? (down ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K) :
                             (down ? GGML_TYPE_IQ4_XS : GGML_TYPE_IQ3_XXS);
            }
        }
        auto * t = z > 1 ? ggml_new_tensor_3d(ctx.get(), type, x, y, z) :
                   y > 1 ? ggml_new_tensor_2d(ctx.get(), type, x, y) : ggml_new_tensor_1d(ctx.get(), type, x);
        ggml_set_name(t, name.c_str());
        values.emplace_back(ggml_nelements(t));
        const bool norm = name.find("norm.weight") != std::string::npos;
        const float scale = y == 1 ? 0.1f : 0.7f/std::sqrt(float(x));
        size_t i = 0;
        for (auto & value : values.back()) {
            rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5;
            value = (float(rng & 65535)/32767.5f-1)*scale + (norm ? 1 : 0);
            // Separate scores, so graph comparisons do not depend on near-ties.
            // The 192-expert kernel fixture separately exercises varied routes.
            if (name.find("exp_probs_b") != std::string::npos) value = float(i)*0.2f-1.5f;
            ++i;
        }
        if (type == GGML_TYPE_F32) t->data = values.back().data();
        else {
            packed.emplace_back(ggml_nbytes(t));
            std::vector<float> importance(x, 1);
            if (ggml_quantize_chunk(type, values.back().data(), packed.back().data(), 0, y*z, x, importance.data()) != packed.back().size())
                throw std::runtime_error("Hy3 fixture quantization failed");
            t->data = packed.back().data();
        }
        gguf_add_tensor(f, t);
    };
    add("token_embd.weight",256,64); add("output.weight",256,64); add("output_norm.weight",256);
    for (int il=0; il<3; ++il) {
        const std::string p="blk."+std::to_string(il)+".";
        add(p+"attn_norm.weight",256); add(p+"ffn_norm.weight",256);
        add(p+"attn_q_norm.weight",128); add(p+"attn_k_norm.weight",128);
        add(p+"attn_q.weight",256,512); add(p+"attn_k.weight",256,128); add(p+"attn_v.weight",256,128);
        add(p+"attn_output.weight",512,256);
        if (il==0) {
            add(p+"ffn_gate.weight",256,512); add(p+"ffn_up.weight",256,512); add(p+"ffn_down.weight",512,256);
        } else {
            add(p+"ffn_gate_inp.weight",256,16); add(p+"exp_probs_b",16);
            for (const std::string kind : {"gate","up","down"}) {
                add(p+"ffn_"+kind+"_exps.weight",256,256,16);
                add(p+"ffn_"+kind+"_shexp.weight",256,256);
            }
        }
    }
    add("blk.2.nextn.eh_proj.weight",512,256);
    add("blk.2.nextn.enorm.weight",256); add("blk.2.nextn.hnorm.weight",256);
    add("blk.2.nextn.shared_head_norm.weight",256);
    if (!gguf_write_to_file(f,path.c_str(),false)) throw std::runtime_error("cannot write Hy3 fixture");
}
