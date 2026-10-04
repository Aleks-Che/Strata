// Validate the complete scheduled graph before copying weights or executing it.
if (strata_glm_sync_active) {
    uint64_t gpu_nodes = 0, expert_nodes = 0;
    for (int si = 0; si < sched->n_splits; ++si) {
        const auto & sp = sched->splits[si];
        auto * dev = ggml_backend_get_device(sched->backends[sp.backend_id]);
        const bool gpu = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
        for (int ni = 0; ni < sp.graph.n_nodes; ++ni) {
            const auto * node = sp.graph.nodes[ni];
            if (node->op == GGML_OP_NONE || ggml_is_view_op(node->op)) continue;
            if (!gpu) {
                ++strata_glm_sync_counters.rejected_cpu_nodes;
                GGML_LOG_ERROR("STRATA_GLM_GPU_ONLY rejected %s (%s) on %s before compute\n",
                    node->name, ggml_op_name(node->op), ggml_backend_name(sched->backends[sp.backend_id]));
                return GGML_STATUS_FAILED;
            }
            ++gpu_nodes;
            if (node->op == GGML_OP_MUL_MAT_ID) ++expert_nodes;
        }
    }
    ++strata_glm_sync_counters.compute_calls;
    strata_glm_sync_counters.gpu_nodes += gpu_nodes;
    strata_glm_sync_counters.expert_nodes += expert_nodes;
}
