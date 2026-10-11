                bool strata_ordered_weight=false;
                // Multiple scheduler copies already use GPU event waits and
                // keep their original dependency protocol.
                if(strata_ordered_enabled && sched->n_copies==1 && split->graph.n_nodes>0 &&
                   ggml_backend_buffer_get_usage(input->buffer)==GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                   ggml_backend_buffer_is_host(input->buffer)) {
                    const auto *consumer=split->graph.nodes[0];
                    if(consumer->op==GGML_OP_MUL_MAT_ID && consumer->src[0]==input_cpy) {
                        auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(split_backend));
                        auto ordered=(StrataExpertOrdered)ggml_backend_reg_get_proc_address(reg,"strata_expert_ordered");
                        strata_ordered_weight=ordered && ordered(split_backend,consumer);
                    }
                }
