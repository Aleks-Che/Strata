                    bool strata_queue_prepared=false;
                    if(ids_backend==split_backend && ids_tensor->op!=GGML_OP_GET_ROWS && strata_pipeline_enabled &&
                       std::getenv("STRATA_DS4_GPU_FETCH_QUEUE") && std::atoi(std::getenv("STRATA_DS4_GPU_FETCH_QUEUE"))) {
                        auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(split_backend));
                        auto fetch=(StrataExpertGpuFetch)ggml_backend_reg_get_proc_address(reg,"strata_expert_gpu_fetch");
                        strata_sched_profile.tokens=ids_tensor->ne[1];
                        auto submitted=[&] {
                            const bool ok=fetch && fetch(split_backend,input,node,ids_tensor);
                            if(ok && strata_sched_profile.file)++strata_sched_profile.samples[StrataSchedulerProfile::gpu_fetch].calls;
                            return ok;
                        };
                        if(submitted())continue;
                        if(ids_tensor!=prev_ids_tensor) {
                            strata_ds4::CpuRouteProfile::begin(ids_tensor);
#define STRATA_CPU_QUEUE_PREPARE
#include "cpu_expert_plan_sched.inc"
#undef STRATA_CPU_QUEUE_PREPARE
                            // A declined empty snapshot must not be prepared a
                            // second time at the legacy route-read boundary.
                            strata_queue_prepared=true;
                            if(whole_ready && submitted())continue;
                        }
                    }
