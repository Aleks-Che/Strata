                    if(strata_ordered_weight && sched->n_copies==1 && !sched->callback_eval &&
                       std::getenv("STRATA_DS4_EARLY_HITS")) {
                        // Do not read an activation copy whose transfer has not
                        // been submitted yet. The route below is the original
                        // tensor just read by the scheduler, not a stale copy.
                        auto *consumer=split->graph.nodes[0];
                        bool ready=true;
                        for(int j=input_id;j<split->n_inputs;++j)
                            if(consumer->src[1]==tensor_copy(split->inputs[j],split_backend_id,sched->cur_copy))ready=false;
                        // These are the pinned CUDA fusion prefixes for an ID
                        // matvec. Fused nodes keep their normal execution.
                        if(split->graph.n_nodes>1) {
                            const auto next=split->graph.nodes[1]->op;
                            if(next==GGML_OP_MUL_MAT_ID || next==GGML_OP_ADD_ID || next==GGML_OP_RESHAPE)ready=false;
                        }
                        if(ready) {
                            auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(split_backend));
                            auto early=(StrataExpertEarlyHits)ggml_backend_reg_get_proc_address(reg,"strata_expert_early_hits");
                            if(early) {
                                int32_t selected[256];size_t count=0;
                                for(int e=0;e<n_expert && e<256;++e)if(ggml_bitset_get(used_ids.data(),e))selected[count++]=e;
                                early(split_backend,input,consumer,ids_tensor,selected,count);
                            }
                        }
                    }
