// Plan gate/up/down in actual split order after reading this route's true IDs.
if(hy3_pipeline && hy3_sync_mode==2) {
    std::vector<Hy3PlanInput> plan;
    for(int next=split_id;next<sched->n_splits;++next) {
        const auto & future=splits[next];
        if(future.backend_id!=split_backend_id || !future.graph.n_nodes) continue;
        auto * op=future.graph.nodes[0];
        if(op->op!=GGML_OP_MUL_MAT_ID) continue;
        auto * route=op->src[2];
        for(int j=0;j<future.n_inputs;++j)
            if(route==tensor_copy(future.inputs[j],split_backend_id,sched->cur_copy)) {route=future.inputs[j];break;}
        if(route!=ids_tensor) break;
        for(int j=0;j<future.n_inputs;++j) {
            auto * weight=future.inputs[j];
            if(op->src[0]!=tensor_copy(weight,split_backend_id,sched->cur_copy) ||
                !ggml_backend_buffer_is_host(weight->buffer) ||
                ggml_backend_buffer_get_usage(weight->buffer)!=GGML_BACKEND_BUFFER_USAGE_WEIGHTS) continue;
            if(weight->ne[2]!=n_expert) return GGML_STATUS_FAILED;
            for(int e=0;e<n_expert;++e) if(ggml_bitset_get(used_ids.data(),e)) plan.push_back({weight,e});
        }
    }
    if(!hy3_begin_plan(plan,ids_tensor->ne[1]==1)) return GGML_STATUS_FAILED;
}
