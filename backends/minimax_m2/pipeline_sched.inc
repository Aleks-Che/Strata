// Plan the actual next matrices for the observed route/layer, never future IDs.
if(mm27_mode==2 && mm27_pipeline_readers) {
    auto ranges_for=[&](const ggml_tensor *weight) {
        mm27_ranges ranges;const auto stride=weight->nb[2];
        for(int64_t e=0;e<n_expert;) {
            if(!ggml_bitset_get(used_ids.data(),e)) {++e;continue;}
            const auto first=e++;
            while(e<n_expert && ggml_bitset_get(used_ids.data(),e))++e;
            ranges.push_back({size_t(first)*stride,size_t(e-first)*stride+(e<n_expert?std::min<size_t>(stride,512):0)});
        }
        return ranges;
    };
    const auto ranges=ranges_for(input);
    if(!mm27_pipeline_lookahead) {
        if(!mm27_pipeline_tensor(split_backend,input_cpy,input,ranges))return GGML_STATUS_FAILED;
    } else {
        if(!mm27_pipeline_active || mm27_plan_next==mm27_plan.size()) {
            std::vector<mm27_plan_tensor> plan;
            const std::string name=input->name;const auto layer=name.substr(0,name.find(".ffn_"));
            for(int next=split_id;next<sched->n_splits && next<split_id+32 && plan.size()<3;++next) {
                const auto &future=splits[next];
                if(future.backend_id!=split_backend_id)break;
                if(!future.graph.n_nodes)continue;
                const auto *op=future.graph.nodes[0];if(op->op!=GGML_OP_MUL_MAT_ID)continue;
                auto *route=op->src[2];
                for(int j=0;j<future.n_inputs;++j)
                    if(route==tensor_copy(future.inputs[j],split_backend_id,sched->cur_copy)) {route=future.inputs[j];break;}
                if(route!=ids_tensor)break;
                const ggml_tensor *weight=nullptr;
                for(int j=0;j<future.n_inputs;++j)if(op->src[0]==tensor_copy(future.inputs[j],split_backend_id,sched->cur_copy)) {
                    weight=future.inputs[j];break;
                }
                if(!weight || !mm27_expert(weight) || weight->ne[2]!=n_expert ||
                    !ggml_backend_buffer_is_host(weight->buffer) ||
                    ggml_backend_buffer_get_usage(weight->buffer)!=GGML_BACKEND_BUFFER_USAGE_WEIGHTS)break;
                const std::string other=weight->name;
                if(other.substr(0,other.find(".ffn_"))!=layer)break;
                plan.push_back({weight,ranges_for(weight)});
            }
            if(plan.empty() || plan.front().source!=input)return GGML_STATUS_FAILED;
            if(!mm27_pipeline_begin(split_backend,std::move(plan)))return GGML_STATUS_FAILED;
        }
        if(!mm27_pipeline_deliver(split_backend,input_cpy,input,ranges))return GGML_STATUS_FAILED;
    }
    continue;
}
