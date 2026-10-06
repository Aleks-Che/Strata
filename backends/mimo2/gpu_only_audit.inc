// Reject CPU math before copies or computation of this graph.
if(mimo_mode) {
    for(int si=0;si<sched->n_splits;++si) {
        const auto &split=sched->splits[si];
        auto *backend=sched->backends[split.backend_id];
        auto *device=ggml_backend_get_device(backend);
        for(int ni=0;ni<split.graph.n_nodes;++ni) {
            const auto *node=split.graph.nodes[ni];
            if(node->op==GGML_OP_NONE || ggml_is_view_op(node->op))continue;
            if(!device || ggml_backend_dev_type(device)!=GGML_BACKEND_DEVICE_TYPE_GPU) {
                ++mimo_stats.rejected_cpu_nodes;
                GGML_LOG_ERROR("STRATA_MIMO_GPU_ONLY rejected %s (%s) on %s\n",node->name,ggml_op_name(node->op),ggml_backend_name(backend));
                return GGML_STATUS_FAILED;
            }
            ++mimo_stats.gpu_nodes;mimo_stats.expert_nodes+=node->op==GGML_OP_MUL_MAT_ID;
        }
    }
    ++mimo_stats.compute_calls;
}
