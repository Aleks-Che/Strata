            if(sched->n_copies==1 && !sched->callback_eval && split->i_end>split->i_start &&
               std::getenv("STRATA_DS4_COMPACT_EXPERTS") && input->buffer &&
               ggml_backend_buffer_is_host(input->buffer) &&
               ggml_backend_buffer_get_usage(input->buffer)==GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                auto *consumer=graph->nodes[split->i_start];
                bool sole=consumer->op==GGML_OP_MUL_MAT_ID && consumer->src[0]==input_cpy;
                for(int n=0;sole && n<graph->n_nodes;++n)for(int s=0;s<GGML_MAX_SRC;++s)
                    if(graph->nodes[n]->src[s]==input_cpy && (graph->nodes[n]!=consumer || s!=0))sole=false;
                if(split->i_end>split->i_start+1) {
                    const auto next=graph->nodes[split->i_start+1]->op;
                    if(next==GGML_OP_MUL_MAT_ID || next==GGML_OP_ADD_ID || next==GGML_OP_RESHAPE)sole=false;
                }
                if(sole) {
                    auto backend=sched->backends[split->backend_id];
                    auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
                    auto compact=(StrataExpertCompact)ggml_backend_reg_get_proc_address(reg,"strata_expert_compact");
                    int slots=compact?compact(backend,consumer):0;
                    if(slots && size_t(slots)*(input_cpy->nb[2]+512)<ggml_nbytes(input_cpy)) {
                        input_cpy->op_params[0]=0x53444350;input_cpy->op_params[1]=slots;
                    }
                }
            }
