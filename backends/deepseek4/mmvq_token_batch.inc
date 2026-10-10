// Quantization already covers all input rows. Reinterpret tokens as samples
// of the single-token kernel; preserve its warp count and reduction order.
// Only these three formats are enabled/tested for DeepSeek Q3 + DSpark.
if constexpr (!strata_token_batch &&
              (type==GGML_TYPE_IQ3_XXS || type==GGML_TYPE_MXFP4 || type==GGML_TYPE_Q8_0)) {
    const int mode=strata_ds4_mmvq_mode();
    const bool fused=fusion.gate || fusion.x_bias || fusion.gate_bias || fusion.x_scale || fusion.gate_scale;
    if ((mode==2 || (mode==1 && ids)) && !fused && ncols_dst>=2 && ncols_dst<=4 &&
        nsamples_x==1 && nsamples_dst==1 && nrows_x%32==0 &&
        (ids || (nchannels_x==1 && nchannels_y==1 && nchannels_dst==1)) &&
        GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        if(std::getenv("STRATA_DS4_MMVQ_TRACE")) {
            static bool seen[5][2]={};
            if(!seen[ncols_dst][ids!=nullptr]) {
                seen[ncols_dst][ids!=nullptr]=true;
                GGML_LOG_INFO("STRATA_MMVQ_TOKEN_BATCH type=%s tokens=%d routed=%d k=%d m=%d\n",
                    ggml_type_name(type),ncols_dst,int(ids!=nullptr),ncols_x,nrows_x);
            }
        }
        mul_mat_vec_q_switch_ncols_dst<type,true>(
            vx,vy,ids,fusion,dst,ncols_x,nrows_x,1,
            stride_row_x,stride_col_y,stride_col_dst,
            nchannels_x,nchannels_y,nchannels_dst,
            stride_channel_x,stride_channel_y,stride_channel_dst,
            1,ncols_dst,0,stride_col_y,stride_col_dst,ids_stride,stream);
        return;
    }
}
