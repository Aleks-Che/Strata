// Included in switch_ncols_dst before its fastdiv/launch parameters are built.
// Quantization has already covered every input row. Interpret the token axis
// as independent samples, retaining the exact ncols=1 reduction configuration.
// The token_batch kernel specialization adjusts only routed ID addressing.
if constexpr (!strata_token_batch) {
    const bool fused = fusion.gate || fusion.x_bias || fusion.gate_bias ||
                       fusion.x_scale || fusion.gate_scale;
    if (strata_mmvq_token_batch_enabled() && !fused && ncols_dst >= 2 && ncols_dst <= 4 &&
        nsamples_x == 1 && nsamples_dst == 1 &&
        (ids || (nchannels_x == 1 && nchannels_y == 1 && nchannels_dst == 1))) {
        mul_mat_vec_q_switch_ncols_dst<type, true>(
            vx, vy, ids, fusion, dst, ncols_x, nrows_x, 1,
            stride_row_x, stride_col_y, stride_col_dst,
            nchannels_x, nchannels_y, nchannels_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            1, ncols_dst, 0, stride_col_y, stride_col_dst, ids_stride, stream);
        return;
    }
}
