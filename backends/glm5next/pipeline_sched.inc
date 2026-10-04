// Router IDs are already on the host. Plan only weights sharing these exact
// routing IDs, in their future scheduler consumption order (gate/up/down).
if (strata_glm_plan_callback && strata_glm_sync_active && !strata_glm_candidate_copy) {
    std::vector<strata_glm_expert> plan;
    for (int next=split_id;next<sched->n_splits;++next) {
        const auto & future=splits[next];
        if (future.backend_id!=split_backend_id || future.graph.n_nodes==0) continue;
        auto * op=future.graph.nodes[0];
        if (op->op!=GGML_OP_MUL_MAT_ID) continue;
        auto * route=op->src[2];
        for (int j=0;j<future.n_inputs;++j)
            if (route==tensor_copy(future.inputs[j],split_backend_id,sched->cur_copy)) {route=future.inputs[j];break;}
        if (route!=ids_tensor) break;
        for (int j=0;j<future.n_inputs;++j) {
            auto * weight=future.inputs[j];
            if (op->src[0]!=tensor_copy(weight,split_backend_id,sched->cur_copy) ||
                !ggml_backend_buffer_is_host(weight->buffer) ||
                ggml_backend_buffer_get_usage(weight->buffer)!=GGML_BACKEND_BUFFER_USAGE_WEIGHTS) continue;
            GGML_ASSERT(weight->ne[2]==n_expert);
            for (int e=0;e<n_expert;++e)
                if (ggml_bitset_get(used_ids.data(),e)) plan.push_back({weight,e});
        }
    }
    strata_glm_plan_callback(strata_glm_copy_owner,plan.data(),plan.size(),ids_tensor->ne[1]<=8,&strata_glm_sync_counters);
}
