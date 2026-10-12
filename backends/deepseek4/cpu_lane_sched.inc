                    if(strata_ordered_weight && sched->n_copies==1 && !sched->callback_eval &&
                       std::getenv("STRATA_DS4_CPU_THREADS") && std::atoi(std::getenv("STRATA_DS4_CPU_THREADS"))) {
                        auto *node=split->graph.nodes[0];bool ready=true;
                        for(int j=input_id;j<split->n_inputs;++j)
                            if(node->src[1]==tensor_copy(split->inputs[j],split_backend_id,sched->cur_copy))ready=false;
                        if(split->graph.n_nodes>1) {
                            const auto next=split->graph.nodes[1]->op;
                            if(next==GGML_OP_MUL_MAT_ID || next==GGML_OP_ADD_ID || next==GGML_OP_RESHAPE)ready=false;
                        }
                        if(ready || !std::getenv("STRATA_DS4_CPU_MATRIX_LEGACY")) {
                            auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(split_backend));
                            auto begin=(StrataExpertCpuBegin)ggml_backend_reg_get_proc_address(reg,"strata_expert_cpu_begin");
                            if(begin)begin(split_backend,input,node,ids_tensor,ids.data(),ids.size()*sizeof(int32_t));
                        }
                    }
