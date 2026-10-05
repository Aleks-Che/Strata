// Included in graph_mtp after the input projection and attention norm.
// Inspired by pwilkin's llama.cpp PR #29928 (head 1e24094d7), adapted to the
// pinned Unsloth graph. This NextN has one DSA layer and no recurrent layers:
// its persistent state depends only on the layer input, not its FFN/output.
if (cache_only) {
    auto * attn = inp_mem->get_attn();

    // set_input still fills these inputs even though no attention is evaluated.
    // Keep them allocated; pruning their buffers would leave dangling inputs.
    ggml_build_forward_expand(gf, attn->get_kq_mask());
    if (inp_kp) {
        for (auto * input : {inp_kp->pool_cells, inp_kp->pool_bias,
                            inp_kp->pool_reps, inp_kp->sel_mask, inp_kp->cand_mask}) {
            if (input) ggml_build_forward_expand(gf, input);
        }
        // Null query stops after key/gate AND completed pooled-key writes.
        // Disabling scoring instead would leave long-context pooled keys stale.
        build_indexer(layer, inp_kp, cur, nullptr, indexer_scoring, il);
    }

    auto * kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, hparams.n_lora_kv, 1, n_tokens);
    cb(kv, "mtp_cache_kv", il);
    ggml_build_forward_expand(gf, attn->mctx->cpy_k(ctx0, kv, attn->get_k_idxs(), il));

    // No caller requested logits or hidden rows. The context supports null
    // outputs, and n_outputs + extraction flags participate in graph reuse.
    res->t_logits = nullptr;
    res->t_h_nextn = nullptr;
    return;
}
