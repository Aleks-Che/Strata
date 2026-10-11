    // Treat tokens as independent samples of the single-column F32 MMVF.
    // Only metadata changes: shared weights, no packing or per-token host loop.
    // Modes 2/3 reuse weights across 4/8 columns, with one extra tail launch.
    const char * strata_hc = std::getenv("STRATA_DS4_HC_MMVF");
    const int strata_hc_columns = strata_hc && std::strcmp(strata_hc,"2")==0 ? 4 :
                                 strata_hc && std::strcmp(strata_hc,"3")==0 ? 8 : 1;
    if (strata_hc && (std::strcmp(strata_hc, "1") == 0 || strata_hc_columns > 1) && GGML_CUDA_CC_IS_NVIDIA(cc) &&
        src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[0] == 16384 && src0->ne[1] == 24 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[2] == 1 && src1->ne[3] == 1 && ne11 >= 1 && ne11 <= 4096 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        (strata_projection_suffix(src0->name, ".hc_attn_fn.weight") ||
         strata_projection_suffix(src0->name, ".hc_ffn_fn.weight")) &&
        ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, 1)) {
        strata_projection_trace(src0, src1, dst, "hc_sample_mmvf");
        strata_projection_sample_mmvf(ctx, src0, src1, dst, strata_hc_columns);
        return;
    }

    // Research-only long router; keep the separate short-router switch intact.
    // BF16 weights, F32 activations/accumulation, same reduction at every N.
    const char * strata_router = std::getenv("STRATA_DS4_ROUTER_SAMPLE_MMVF");
    const int strata_router_columns = strata_router && std::strcmp(strata_router,"2")==0 ? 4 :
                                     strata_router && std::strcmp(strata_router,"3")==0 ? 8 : 1;
    if (strata_router && (std::strcmp(strata_router,"1")==0 || strata_router_columns > 1) &&
        GGML_CUDA_CC_IS_NVIDIA(cc) && src0->type==GGML_TYPE_BF16 &&
        src1->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 && dst->op_params[0]==GGML_PREC_F32 &&
        src0->ne[0]==4096 && src0->ne[1]==256 && src0->ne[2]==1 && src0->ne[3]==1 &&
        src1->ne[2]==1 && src1->ne[3]==1 && ne11>=1 && ne11<=4096 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        strata_projection_suffix(src0->name,".ffn_gate_inp.weight") &&
        ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, 1)) {
        strata_projection_trace(src0, src1, dst, "router_sample_mmvf");
        strata_projection_sample_mmvf(ctx, src0, src1, dst, strata_router_columns);
        return;
    }

    // Keep gate/up, and optionally down, on the same Q8 scale and K partition.
    if (strata_shared_mmq_eligible(src0,src1,dst,cc)) {
        strata_projection_trace(src0,src1,dst,"shared_fixed_mmq");
        strata_projection_fixed_mmq(ctx,src0,src1,nullptr,dst);
        return;
    }

    // Mode 1 isolates the short-batch MMVQ -> MMQ boundary. Mode 2 also fixes
    // the tile/K partition for batches up to 4096. Mode 3 adds Q_b;
    // mode 4 also handles the eight grouped output-A projections; mode 5 adds B;
    // mode 6 fixes the CSA/HCA and indexer compressor KV/gate projections.
    const char * strata_qk_mmq = std::getenv("STRATA_DS4_PROJECTION_MMQ");
    const bool strata_comp_mode = strata_qk_mmq && std::strcmp(strata_qk_mmq, "6") == 0;
    const bool strata_ob_mode = strata_qk_mmq && (std::strcmp(strata_qk_mmq, "5") == 0 || strata_comp_mode);
    const bool strata_oa_mode = strata_qk_mmq && (std::strcmp(strata_qk_mmq, "4") == 0 || strata_ob_mode);
    const bool strata_qb_mode = strata_qk_mmq && (std::strcmp(strata_qk_mmq, "3") == 0 || strata_oa_mode);
    const bool strata_fixed_qk = strata_qk_mmq && (std::strcmp(strata_qk_mmq, "2") == 0 || strata_qb_mode);
    const bool strata_qb_shape = strata_qb_mode && src0->ne[0]==1024 && src0->ne[1]==32768 &&
        strata_projection_suffix(src0->name,".attn_q_b.weight");
    const bool strata_ob_shape = strata_ob_mode && src0->ne[0]==8192 && src0->ne[1]==4096 &&
        strata_projection_suffix(src0->name,".attn_output_b.weight");
    // The model permutes [K,groups,tokens] to [K,tokens,groups]. Dense MMQ
    // already quantizes with all three input strides; no packing copy needed.
    const bool strata_oa_shape = strata_oa_mode && src0->ne[0]==4096 && src0->ne[1]==1024 &&
        src0->ne[2]==8 && src0->ne[3]==1 && src1->ne[2]==8 && src1->ne[3]==1 &&
        src1->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 &&
        strata_projection_suffix(src0->name,".attn_output_a.weight");
    const bool strata_oa_layout = strata_oa_shape && (ggml_is_contiguous(src1) ||
        (src1->nb[0]==sizeof(float) && src1->nb[2]==4096*sizeof(float) &&
         src1->nb[1]==8*src1->nb[2] && src1->nb[3]==size_t(ne11)*src1->nb[1]));
    const bool strata_comp_shape = strata_comp_mode && src0->ne[0]==4096 &&
        src1->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 &&
        (((src0->ne[1]==512 || src0->ne[1]==1024) &&
          (strata_projection_suffix(src0->name,".attn_compressor_kv.weight") ||
           strata_projection_suffix(src0->name,".attn_compressor_gate.weight"))) ||
         (src0->ne[1]==256 &&
          (strata_projection_suffix(src0->name,".indexer_compressor_kv.weight") ||
           strata_projection_suffix(src0->name,".indexer_compressor_gate.weight"))));
    const bool strata_dense_shape = strata_qb_shape || strata_ob_shape || strata_comp_shape ||
        (src0->ne[0]==4096 && (src0->ne[1]==1024 || src0->ne[1]==512) && strata_projection_qk(src0));
    const bool strata_dense_layout = src0->ne[2]==1 && src0->ne[3]==1 &&
        src1->ne[2]==1 && src1->ne[3]==1 && ggml_is_contiguous(src1);
    if (strata_qk_mmq && (std::strcmp(strata_qk_mmq, "1") == 0 || strata_fixed_qk) &&
        GGML_CUDA_CC_IS_NVIDIA(cc) && src0->type == GGML_TYPE_Q8_0 &&
        (!strata_fixed_qk || cc >= GGML_CUDA_CC_AMPERE) &&
        (strata_oa_layout || (strata_dense_shape && strata_dense_layout)) &&
        ne11 >= 1 && ne11 <= (strata_fixed_qk ? 4096 : 8) &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(dst) &&
        ggml_cuda_should_use_mmq(src0->type, cc, ne11, 0)) {
        strata_projection_trace(src0, src1, dst, strata_fixed_qk ? "fixed_projection_mmq" : "projection_mmq");
        if (strata_fixed_qk)strata_projection_fixed_mmq(ctx, src0, src1, nullptr, dst);
        else ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
