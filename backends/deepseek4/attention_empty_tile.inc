    // Only the private single-query sparse partition is eligible. Preserve
    // the grid and every nonempty tile's reduction. Empty trailing partitions
    // contribute +0, a zero row sum, and the unchanged initial max sentinel.
    if constexpr (strata_skip_empty && use_sparse && DKQ == 512 && DV == 512 && ncols1 == 1 && ncols2 == 8 &&
                  (needs_fixup || is_fixup)) {
        // Leave the first partition on the original path. The last partition
        // (not is_fixup) owns the sink, even when it has no visible K/V rows.
        if (kb0_start > 0 && kb0_start >= kb0_stop) {
            const int tid = threadIdx.y * blockDim.x + threadIdx.x;
            const int threads = blockDim.x * blockDim.y;
            float2 *meta = dstk_fixup + (is_fixup ? gridDim.x + blockIdx.x : blockIdx.x) * ncols;
            if (tid < ncols) {
                const float maximum = !is_fixup && sinks_f ? fmaxf(-FLT_MAX/2.0f, sinks_f[tid]) : -FLT_MAX/2.0f;
                const float rowsum = !is_fixup && sinks_f ? expf(sinks_f[tid] - maximum) : 0.0f;
                meta[tid] = make_float2(maximum, rowsum);
            }
            for (int i = tid; i < ncols * (DV/2); i += threads) {
                const int c = i / (DV/2), d = i % (DV/2);
                if constexpr (is_fixup) {
                    dstk_fixup[gridDim.x * (2*ncols) + blockIdx.x * (ncols*(DV/2)) + i] = make_float2(0.0f, 0.0f);
                } else if (zt_gqa*ncols2 + c < gqa_ratio) {
                    dstk[(jt*ne02 + c)*(DV/2) + d] = make_float2(0.0f, 0.0f);
                }
            }
            return;
        }
    }
