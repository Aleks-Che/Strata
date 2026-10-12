    // Per-execution metadata: graph reuse must never reuse token IDs or plans.
    struct StrataHashGroup {
        ggml_tensor *route=nullptr;
        ggml_backend_t backend=nullptr;
        int last_split=-1;
        std::vector<ggml_tensor *> weights;
        std::vector<StrataExpertSlice> slices;
        std::vector<int32_t> ids;
    };
    std::vector<StrataHashGroup> strata_hash_groups;
    StrataExpertEarlyPlan strata_early_plan=nullptr;
    const char *strata_hash_env=std::getenv("STRATA_DS4_HASH_PREFETCH");
    const bool strata_hash_fast=strata_hash_env && std::strcmp(strata_hash_env,"2")==0;
    if(strata_ordered_enabled && sched->n_copies==1 && !sched->callback_eval &&
       strata_hash_env && (std::strcmp(strata_hash_env,"1")==0 || strata_hash_fast)) {
        // Scheduler rewrites sources to input copies. At graph entry those
        // copies may still contain the previous execution's tokens.
        auto original=[&](ggml_tensor *tensor) {
            for(int s=0;s<sched->n_splits;++s)for(int j=0;j<splits[s].n_inputs;++j)
                if(tensor==tensor_copy(splits[s].inputs[j],splits[s].backend_id,sched->cur_copy))
                    return splits[s].inputs[j];
            return tensor;
        };
        ggml_tensor *tokens=nullptr;
        StrataHashRoutes routes;
        for(int s=0;s<sched->n_splits;++s) {
            auto &future=splits[s];
            if(!future.graph.n_nodes)continue;
            auto *op=future.graph.nodes[0];
            if(op->op!=GGML_OP_MUL_MAT_ID)continue;
            ggml_tensor *weight=nullptr;
            for(int j=0;j<future.n_inputs;++j) {
                auto *w=future.inputs[j];
                if(op->src[0]==tensor_copy(w,future.backend_id,sched->cur_copy) && w->buffer &&
                   ggml_backend_buffer_is_host(w->buffer) &&
                   ggml_backend_buffer_get_usage(w->buffer)==GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    weight=w;break;
                }
            }
            if(!weight)continue;
            auto *route=original(op->src[2]);
            auto *backend=sched->backends[future.backend_id];
            if(strata_hash_groups.empty() || route!=strata_hash_groups.back().route) {
                if(routes.count==3 || route->op!=GGML_OP_GET_ROWS || route->type!=GGML_TYPE_I32 ||
                   route->ne[0]<1 || route->ne[0]>8 || route->ne[1]<1 || route->ne[1]>8 ||
                   route->ne[2]!=1 || route->ne[3]!=1 || !ggml_is_contiguous(route))break;
                auto *input_tokens=original(route->src[1]);
                auto *table=original(route->src[0]);
                if(!input_tokens || input_tokens->op!=GGML_OP_NONE || !(input_tokens->flags&GGML_TENSOR_FLAG_INPUT) ||
                   input_tokens->type!=GGML_TYPE_I32 || !ggml_is_contiguous(input_tokens) ||
                   ggml_nelements(input_tokens)!=route->ne[1] || !input_tokens->buffer || !input_tokens->data ||
                   !table || table->ne[0]!=route->ne[0] ||
                   (tokens && (tokens!=input_tokens || backend!=strata_hash_groups.front().backend)))break;
                tokens=input_tokens;
                routes.n_tokens=int(route->ne[1]);routes.tables[routes.count++]=table;
                strata_hash_groups.push_back({route,backend,s,{},{},{}});
            }
            auto &group=strata_hash_groups.back();
            if(backend!=group.backend || !weight->data || weight->ne[2]<1 || weight->ne[2]>256 ||
               weight->ne[3]!=1 || !ggml_is_contiguous(weight)) {
                strata_hash_groups.clear();break;
            }
            group.weights.push_back(weight);group.last_split=s;
        }
        if(!strata_hash_groups.empty()) {
            auto *backend=strata_hash_groups.front().backend;
            auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            auto read=(StrataExpertHashRead)ggml_backend_reg_get_proc_address(reg,"strata_expert_hash_read");
            strata_early_plan=(StrataExpertEarlyPlan)ggml_backend_reg_get_proc_address(reg,"strata_expert_early_plan");
            // Reading original input metadata works with both CPU and GPU input
            // buffers; the lookup itself runs only on the CUDA backend.
            if(read && strata_early_plan)ggml_backend_tensor_get(tokens,routes.tokens,0,routes.n_tokens*sizeof(int32_t));
            bool valid=read && strata_early_plan && read(backend,&routes);
            for(size_t i=0;valid && i<strata_hash_groups.size();++i) {
                auto &group=strata_hash_groups[i];
                group.ids.assign(routes.ids[i],routes.ids[i]+group.route->ne[0]*routes.n_tokens);
                for(auto *weight:group.weights) {
                    bool used[256]={};
                    for(int j=0;j<group.route->ne[0]*routes.n_tokens;++j) {
                        int e=routes.ids[i][j];
                        if(e<0 || e>=weight->ne[2]) {valid=false;break;}
                        used[e]=true;
                    }
                    if(!valid)break;
                    for(int e=0;e<weight->ne[2];++e)if(used[e]) {
                        size_t width=weight->nb[2];
                        size_t bytes=width+(e<weight->ne[2]-1?std::min<size_t>(width,512):0);
                        group.slices.push_back({(const uint8_t *)weight->data+size_t(e)*width,bytes,true});
                    }
                }
            }
            if(!valid)strata_hash_groups.clear();
        }
    }
    auto strata_start_hash_group=[&](size_t i) {
        auto &group=strata_hash_groups[i];
        strata_early_plan(group.backend,group.slices.data(),group.slices.size());
    };
    if(!strata_hash_groups.empty())strata_start_hash_group(0);
    auto strata_reuse_hash_ids=[&](ggml_backend_t backend,ggml_tensor *route,std::vector<int32_t> &ids) {
        if(!strata_hash_fast)return false;
        for(const auto &group:strata_hash_groups)
            if(group.backend==backend && group.route==route && group.ids.size()==ids.size()) {
                // The original GET_ROWS reads the same immutable GPU table and
                // this execution's input tokens. Reuse its exact early result;
                // graph computation and copies still retain CUDA stream order.
                std::copy(group.ids.begin(),group.ids.end(),ids.begin());return true;
            }
        return false;
    };
