#pragma once
// Native MiMo graph fixture: reduced hidden/FFN/expert count, real Q/K/V geometry.
// Never admitted as a production checkpoint by the Python model contract.
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

inline void write_synthetic_mimo2(const std::string & path, bool mixed, bool dequantized = false) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ggml_tensor_overhead()*128, nullptr, true}), ggml_free);
    if (!file || !ctx) throw std::runtime_error("MiMo fixture allocation failed");
    auto * f = file.get();
    gguf_set_val_str(f, "general.architecture", "mimo2");
    gguf_set_val_str(f, "general.name", "Strata synthetic MiMo full-SWA fixture");
    gguf_set_val_str(f, "tokenizer.ggml.model", "none");
    auto u = [f](const char * k, uint32_t n) { gguf_set_val_u32(f, (std::string("mimo2.")+k).c_str(), n); };
    auto v = [f](const char * k, float n) { gguf_set_val_f32(f, (std::string("mimo2.")+k).c_str(), n); };
    u("block_count", 3); u("context_length", 1024); u("embedding_length", 256);
    u("feed_forward_length", 512); u("vocab_size", 64); u("nextn_predict_layers", 0);
    u("attention.head_count", 64); u("attention.key_length", 192); u("attention.value_length", 128);
    u("attention.sliding_window", 128); v("attention.layer_norm_rms_epsilon", 1e-6f);
    v("attention.value_scale", .707f); u("rope.dimension_count", 64);
    v("rope.freq_base", 1e7f); v("rope.freq_base_swa", 1e4f);
    u("expert_count", 16); u("expert_used_count", 8); u("expert_feed_forward_length", 256);
    u("expert_gating_func", 2); u("expert_group_count", 1); u("expert_group_used_count", 1);
    const uint32_t kv[] = {4, 8, 4}, swa[] = {0, 1, 0};
    gguf_set_arr_data(f, "mimo2.attention.head_count_kv", GGUF_TYPE_UINT32, kv, 3);
    gguf_set_arr_data(f, "mimo2.attention.sliding_window_pattern", GGUF_TYPE_UINT32, swa, 3);
    uint32_t rng = 0x26d001u;
    std::vector<std::vector<float>> values;
    std::vector<std::vector<uint8_t>> packed;
    auto add = [&](const std::string & name, int64_t x, int64_t y = 1, int64_t z = 1) {
        const bool router = name.find("ffn_gate_inp") != std::string::npos;
        ggml_type type = !mixed || y == 1 || router ? GGML_TYPE_F32 : GGML_TYPE_BF16;
        if (mixed && z > 1) type = name.find("_down_") != std::string::npos ? GGML_TYPE_MXFP4 :
                                  name.find("_up_") != std::string::npos ? GGML_TYPE_Q3_K : GGML_TYPE_Q2_K;
        auto * t = ggml_new_tensor_3d(ctx.get(), dequantized ? GGML_TYPE_F32 : type, x, y, z);
        ggml_set_name(t, name.c_str());
        values.emplace_back(ggml_nelements(t));
        const bool norm = name.find("norm.weight") != std::string::npos;
        const float scale = y == 1 && z == 1 ? .1f : .35f/std::sqrt(float(x));
        size_t i = 0;
        for (auto & a : values.back()) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            a = (float(rng & 0xffffu)/32767.5f-1)*scale;
            if (norm) a += 1;
            if (name.find("exp_probs_b.bias") != std::string::npos) a = .2f*float(i)-1.5f;
            if (name.find("attn_sinks") != std::string::npos) a = 1.f + .01f*float(i);
            ++i;
        }
        if (type != GGML_TYPE_F32) {
            const size_t row = ggml_row_size(type, x);
            packed.emplace_back(row*y*z);
            std::vector<float> importance(x, 1.f);
            if (ggml_quantize_chunk(type, values.back().data(), packed.back().data(), 0, y*z, x, importance.data()) != packed.back().size())
                throw std::runtime_error("MiMo fixture quantization failed");
            if (dequantized) {
                const auto * traits = ggml_get_type_traits(type);
                if (!traits->to_float) throw std::runtime_error("missing dequantization reference");
                for (int64_t r = 0; r < y*z; ++r) traits->to_float(packed.back().data()+r*row, values.back().data()+r*x, x);
            }
        }
        t->data = type == GGML_TYPE_F32 || dequantized ? static_cast<void *>(values.back().data()) : packed.back().data();
        gguf_add_tensor(f, t);
    };
    add("token_embd.weight", 256, 64); add("output.weight", 256, 64); add("output_norm.weight", 256);
    for (int layer = 0; layer < 3; ++layer) {
        const std::string p = "blk."+std::to_string(layer)+".";
        add(p+"attn_norm.weight", 256); add(p+"ffn_norm.weight", 256);
        add(p+"attn_qkv.weight", 256, 64*192+kv[layer]*(192+128));
        add(p+"attn_output.weight", 64*128, 256);
        if (swa[layer]) add(p+"attn_sinks.weight", 64);
        if (layer == 0) {
            add(p+"ffn_gate.weight", 256, 512); add(p+"ffn_up.weight", 256, 512); add(p+"ffn_down.weight", 512, 256);
        } else {
            add(p+"ffn_gate_inp.weight", 256, 16); add(p+"exp_probs_b.bias", 16);
            add(p+"ffn_gate_exps.weight", 256, 256, 16);
            add(p+"ffn_up_exps.weight", 256, 256, 16); add(p+"ffn_down_exps.weight", 256, 256, 16);
        }
    }
    if (!gguf_write_to_file(f, path.c_str(), false)) throw std::runtime_error("cannot write MiMo fixture");
}
