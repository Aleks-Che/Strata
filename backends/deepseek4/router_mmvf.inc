    // MMF rounds F32 activations to BF16 for this weight type. MMVF keeps
    // activations in F32 and uses the same reduction for 1..4 columns.
    // Restrict this opt-in experiment to the measured DeepSeek router shape.
    if (src0->type == GGML_TYPE_BF16 && ne00 == 4096 && ne01 == 256 &&
        ne02 == 1 && ne03 == 1 && ne11 >= 2 && ne11 <= 4 &&
        src1->ne[2] == 1 && src1->ne[3] == 1 && src1->nb[0] == sizeof(float) &&
        src1->nb[1] % (2*sizeof(float)) == 0 && ggml_is_contiguous(dst) &&
        dst->op_params[0] == GGML_PREC_F32 && GGML_CUDA_CC_IS_NVIDIA(cc) &&
        ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, 1)) {
        const char *mode = std::getenv("STRATA_DS4_ROUTER_MMVF");
        const char *suffix = std::strstr(src0->name, ".ffn_gate_inp.weight");
        if (mode && std::strcmp(mode, "1") == 0 && suffix &&
            std::strcmp(suffix, ".ffn_gate_inp.weight") == 0) {
            ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
            return;
        }
    }
