#pragma once
// Bounded diagnostics at the actual MUL_MAT dispatch branch. Disabled by
// default, compiled only into the projection research build.
#include <set>
#include <string>
void strata_projection_fixed_mmq(ggml_backend_cuda_context &,const ggml_tensor *,const ggml_tensor *,const ggml_tensor *,ggml_tensor *);
static bool strata_projection_suffix(const char * name, const char * suffix) {
    const size_t n = std::strlen(name), s = std::strlen(suffix);
    return n >= s && std::memcmp(name + n - s, suffix, s) == 0;
}
static bool strata_projection_qk(const ggml_tensor * w) {
    return strata_projection_suffix(w->name, ".attn_q_a.weight") ||
           strata_projection_suffix(w->name, ".attn_kv.weight");
}
// Reinterpret a dense token axis as columns x independent samples. MMVF
// broadcasts the same weights to every sample. No packing or GPU allocation.
static void strata_projection_sample_mmvf(ggml_backend_cuda_context & ctx,
        const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * dst, int columns) {
    const auto launch = [&](int64_t first, int64_t count, int64_t samples) {
        ggml_tensor input = *x, out = *dst;
        input.ne[1] = out.ne[1] = count;
        input.ne[3] = out.ne[3] = samples;
        input.nb[2] = input.nb[3] = size_t(count)*input.nb[1];
        out.nb[2] = out.nb[3] = size_t(count)*out.nb[1];
        input.data = static_cast<char *>(x->data) + size_t(first)*x->nb[1];
        out.data = static_cast<char *>(dst->data) + size_t(first)*dst->nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, w, &input, nullptr, &out);
    };
    const int64_t groups = x->ne[1]/columns, tail = x->ne[1]%columns;
    if (groups) launch(0, columns, groups);
    if (tail) launch(groups*columns, tail, 1);
}
static void strata_projection_trace(const ggml_tensor * w, const ggml_tensor * x,
                                    const ggml_tensor * dst, const char * path) {
    const char * enabled = std::getenv("STRATA_DS4_PROJECTION_TRACE");
    if (!enabled || std::strcmp(enabled, "1") != 0) return;
    // First occurrence of each architecture: layer 0 ordinary, 2 CSA, 3 HCA.
    const bool compressor = strata_projection_suffix(w->name, ".attn_compressor_kv.weight") ||
        strata_projection_suffix(w->name, ".attn_compressor_gate.weight") ||
        strata_projection_suffix(w->name, ".indexer_compressor_kv.weight") ||
        strata_projection_suffix(w->name, ".indexer_compressor_gate.weight");
    if (std::strncmp(w->name, "blk.0.", 6) != 0 && !(compressor &&
        (std::strncmp(w->name, "blk.2.", 6)==0 || std::strncmp(w->name, "blk.3.", 6)==0))) return;
    if (!strata_projection_qk(w) && !strata_projection_suffix(w->name, ".hc_attn_fn.weight") &&
        !strata_projection_suffix(w->name, ".hc_ffn_fn.weight") &&
        !strata_projection_suffix(w->name, ".attn_q_b.weight") &&
        !strata_projection_suffix(w->name, ".attn_output_a.weight") &&
        !strata_projection_suffix(w->name, ".attn_output_b.weight") &&
        !strata_projection_suffix(w->name, ".ffn_gate_inp.weight") &&
        !strata_projection_suffix(w->name, ".ffn_gate_shexp.weight") &&
        !strata_projection_suffix(w->name, ".ffn_up_shexp.weight") &&
        !strata_projection_suffix(w->name, ".ffn_down_shexp.weight") && !compressor) return;
    std::string key = std::string(w->name) + ":" + ggml_type_name(w->type) + ":" +
        std::to_string(w->ne[0]) + ":" + std::to_string(w->ne[1]) + ":" +
        std::to_string(x->ne[1]) + ":" + std::to_string(dst->op_params[0]) + ":" + path;
    static thread_local std::set<std::string> seen;
    if (seen.size() >= 256 || !seen.insert(key).second) return;
    GGML_LOG_INFO("strata_projection_dispatch weight=%s type=%s k=%lld m=%lld tokens=%lld precision=%d path=%s\n",
        w->name, ggml_type_name(w->type), (long long) w->ne[0], (long long) w->ne[1],
        (long long) x->ne[1], dst->op_params[0], path);
}
