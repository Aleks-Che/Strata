    // Compiled only into Strata's separate DeepSeek backend. Shape/view nodes do
    // not perform arithmetic; every executable forward node must be on a GPU.
    int strata_gpu_nodes = 0, strata_expert_nodes = 0;
    for (int si = 0; si < sched->n_splits; ++si) {
        const auto & sp = sched->splits[si];
        auto * dev = ggml_backend_get_device(sched->backends[sp.backend_id]);
        const bool gpu = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
        for (int ni = 0; ni < sp.graph.n_nodes; ++ni) {
            const auto * node = sp.graph.nodes[ni];
            if (node->op == GGML_OP_NONE || ggml_is_view_op(node->op)) continue;
            if (!gpu) {
                GGML_LOG_ERROR("STRATA_GPU_ONLY rejected CPU operation %s (%s) on %s\n",
                    node->name, ggml_op_name(node->op), ggml_backend_name(sched->backends[sp.backend_id]));
                return GGML_STATUS_FAILED;
            }
            ++strata_gpu_nodes;
            if (node->op == GGML_OP_MUL_MAT_ID) ++strata_expert_nodes;
        }
    }
    static int strata_previous_nodes = -1;
    if (strata_previous_nodes != strata_gpu_nodes) {
        GGML_LOG_INFO("STRATA_GPU_AUDIT gpu_nodes=%d cpu_compute_nodes=0 expert_matmuls_gpu=%d\n",
            strata_gpu_nodes, strata_expert_nodes);
        strata_previous_nodes = strata_gpu_nodes;
    }
