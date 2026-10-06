// Inspect all scheduled splits before any tensor copy or model computation.
if (hy3_sync_mode) {
    uint64_t gpu_nodes = 0, expert_nodes = 0;
    for (int si = 0; si < sched->n_splits; ++si) {
        const auto & split = sched->splits[si];
        auto * backend = sched->backends[split.backend_id];
        auto * device = ggml_backend_get_device(backend);
        const bool gpu = device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU;
        for (int ni = 0; ni < split.graph.n_nodes; ++ni) {
            const auto * node = split.graph.nodes[ni];
            if (node->op == GGML_OP_NONE || ggml_is_view_op(node->op)) continue;
            if (!gpu) {
                ++hy3_sync_stats.rejected_cpu_nodes;
                GGML_LOG_ERROR("STRATA_HY3_GPU_ONLY rejected %s (%s) on %s before compute\n",
                    node->name, ggml_op_name(node->op), ggml_backend_name(backend));
                return GGML_STATUS_FAILED;
            }
            ++gpu_nodes;
            expert_nodes += node->op == GGML_OP_MUL_MAT_ID;
        }
    }
    ++hy3_sync_stats.compute_calls;
    hy3_sync_stats.gpu_nodes += gpu_nodes;
    hy3_sync_stats.expert_nodes += expert_nodes;
}
