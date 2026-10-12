                                uint8_t whole_selected[256]={};
                                StrataCpuExpertBranch whole_branch{};
                                bool whole_ready=false;
                                if(strata_ordered_enabled && sched->n_copies==1 && !sched->callback_eval &&
                                   n_expert<=256 && ids_tensor->ne[1]<=8 &&
                                   std::getenv("STRATA_DS4_CPU_THREADS") && std::atoi(std::getenv("STRATA_DS4_CPU_THREADS"))) {
                                    int found=0;bool safe=true;
                                    for(int next=split_id;next<sched->n_splits && found<3;++next) {
                                        auto &future=splits[next];
                                        if(future.backend_id!=split_backend_id || !future.graph.n_nodes)continue;
                                        auto *op=future.graph.nodes[0];if(op->op!=GGML_OP_MUL_MAT_ID)continue;
                                        auto *route=op->src[2];
                                        for(int j=0;j<future.n_inputs;++j)
                                            if(route==tensor_copy(future.inputs[j],split_backend_id,sched->cur_copy))route=future.inputs[j];
                                        if(route!=ids_tensor)break;
                                        if(future.graph.n_nodes>1) {
                                            auto next_op=future.graph.nodes[1]->op;
                                            if(next_op==GGML_OP_MUL_MAT_ID || next_op==GGML_OP_ADD_ID || next_op==GGML_OP_RESHAPE)safe=false;
                                        }
                                        for(int ni=1;ni<future.graph.n_nodes;++ni)for(int si=0;si<GGML_MAX_SRC;++si)
                                            if(future.graph.nodes[ni]->src[si]==op->src[0])safe=false;
                                        for(int j=0;j<future.n_inputs;++j) {
                                            auto *w=future.inputs[j];
                                            if(op->src[0]!=tensor_copy(w,split_backend_id,sched->cur_copy))continue;
                                            if(!ggml_backend_buffer_is_host(w->buffer) ||
                                               ggml_backend_buffer_get_usage(w->buffer)!=GGML_BACKEND_BUFFER_USAGE_WEIGHTS)continue;
                                            whole_branch.nodes[found]=op;whole_branch.weights[found]=w;++found;break;
                                        }
                                    }
                                    if(found==3 && safe) {
                                        auto *act=whole_branch.nodes[2]->src[1];
                                        // DFS visits gate first, whereas some explicitly expanded
                                        // model graphs visit up first. Bind by edges, never names/order.
                                        if(act->op==GGML_OP_GLU && act->src[0]==whole_branch.nodes[0] && act->src[1]==whole_branch.nodes[1]) {
                                            std::swap(whole_branch.nodes[0],whole_branch.nodes[1]);
                                            std::swap(whole_branch.weights[0],whole_branch.weights[1]);
                                        }
                                        if(act->op==GGML_OP_GLU && ggml_get_glu_op(act)==GGML_GLU_OP_SWIGLU_CLAMP)
                                            std::memcpy(&whole_branch.clamp,&act->op_params[3],sizeof(float));
                                        // No pending host activation copy may race the CPU input D2H.
                                        for(int j=input_id;j<split->n_inputs;++j)
                                            if(whole_branch.nodes[0]->src[1]==tensor_copy(split->inputs[j],split_backend_id,sched->cur_copy))safe=false;
                                        auto cpu_plan=(StrataCpuExpertPlan)ggml_backend_reg_get_proc_address(reg,"strata_cpu_expert_plan");
#ifdef STRATA_CPU_QUEUE_PREPARE
                                        if(safe && cpu_plan)whole_ready=cpu_plan(split_backend,&whole_branch,ids_tensor,nullptr,ggml_nbytes(ids_tensor),whole_selected);
#else
                                        if(safe && cpu_plan)whole_ready=cpu_plan(split_backend,&whole_branch,ids_tensor,ids.data(),ids.size()*4,whole_selected);
#endif
                                    }
                                }
