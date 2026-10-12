                        if(strata_pipeline_enabled) {
                            auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(split_backend));
                            auto plan=(StrataExpertPlan)ggml_backend_reg_get_proc_address(reg,"strata_expert_plan");
                            if(plan) {
                                std::vector<StrataExpertSlice> slices;
#include "cpu_expert_plan_sched.inc"
                                int matrices=0;
                                // Routing IDs are already known. Look ahead only
                                // through matrices using these exact IDs. Never
                                // guess the next layer's route or touch its inputs.
                                for(int next=split_id;next<sched->n_splits;++next) {
                                    auto &future=splits[next];
                                    if(future.backend_id!=split_backend_id || future.graph.n_nodes==0)continue;
                                    auto *op=future.graph.nodes[0];
                                    if(op->op!=GGML_OP_MUL_MAT_ID)continue;
                                    auto *route=op->src[2];
                                    for(int j=0;j<future.n_inputs;++j)
                                        if(route==tensor_copy(future.inputs[j],split_backend_id,sched->cur_copy)) {
                                            route=future.inputs[j];break;
                                        }
                                    if(route!=ids_tensor)break;
                                    for(int j=0;j<future.n_inputs;++j) {
                                        auto *weight=future.inputs[j];
                                        if(op->src[0]!=tensor_copy(weight,split_backend_id,sched->cur_copy) ||
                                           !ggml_backend_buffer_is_host(weight->buffer) ||
                                           ggml_backend_buffer_get_usage(weight->buffer)!=GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
                                            continue;
                                        GGML_ASSERT(weight->ne[2]==n_expert);
                                        ++matrices;
                                        uint8_t cpu_selected[256]={};
                                        if(whole_ready)for(auto *w:whole_branch.weights)if(w==weight)std::memcpy(cpu_selected,whole_selected,sizeof(cpu_selected));
                                        if(strata_ordered_enabled && sched->n_copies==1 && !sched->callback_eval &&
                                           n_expert<=256 && std::getenv("STRATA_DS4_CPU_THREADS") &&
                                           std::atoi(std::getenv("STRATA_DS4_CPU_THREADS"))) {
                                            bool eligible=true;
                                            for(int ni=1;ni<future.graph.n_nodes;++ni)
                                                for(int si=0;si<GGML_MAX_SRC;++si)
                                                    if(future.graph.nodes[ni]->src[si]==op->src[0])eligible=false;
                                            if(future.graph.n_nodes>1) {
                                                const auto following=future.graph.nodes[1]->op;
                                                if(following==GGML_OP_MUL_MAT_ID || following==GGML_OP_ADD_ID || following==GGML_OP_RESHAPE)eligible=false;
                                            }
                                            for(int k=j;k<future.n_inputs;++k)
                                                if(op->src[1]==tensor_copy(future.inputs[k],split_backend_id,sched->cur_copy))eligible=false;
                                            auto select=(StrataExpertCpuSelect)ggml_backend_reg_get_proc_address(reg,"strata_expert_cpu_select");
                                            if(eligible && select)select(split_backend,weight,op,ids_tensor,ids.data(),ids.size()*sizeof(int32_t),cpu_selected);
                                        }
                                        for(int e=0;e<n_expert;++e)if(ggml_bitset_get(used_ids.data(),e)) {
                                            size_t width=weight->nb[2];
                                            size_t bytes=width+(e<n_expert-1?std::min<size_t>(width,512):0);
                                            slices.push_back({(const uint8_t *)weight->data+size_t(e)*width,bytes,ids_tensor->ne[1]<=8,e<256 && cpu_selected[e]!=0});
                                        }
                                    }
                                }
                                plan(split_backend,slices.data(),slices.size());
                                if(std::getenv("STRATA_EXPERT_PIPELINE_TRACE"))
                                    GGML_LOG_INFO("STRATA_EXPERT_PLAN matrices=%d slices=%zu first=%s\n",matrices,slices.size(),input->name);
                            }
                        }
