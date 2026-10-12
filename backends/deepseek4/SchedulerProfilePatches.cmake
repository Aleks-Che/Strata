# Private ordered-copy experiment and optional host timings. Apply after the
# expert copy and pipeline patches; preserve the pinned scheduler source.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/ordered_expert_sched.inc")
function(strata_scheduler_replace old new)
  string(FIND "${scheduler}" "${old}" pos)
  if(pos EQUAL -1)
    message(FATAL_ERROR "Pinned scheduler profiling anchor missing: ${old}")
  endif()
  string(REPLACE "${old}" "${new}" patched "${scheduler}")
  set(scheduler "${patched}" PARENT_SCOPE)
endfunction()
strata_scheduler_replace("${audit_anchor}" "${audit_anchor}\n    StrataSchedulerProfile strata_sched_profile;")
set(destination_wait "                // wait for the split backend to finish using the input before overwriting it\n                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {\n                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);\n                } else {\n                    ggml_backend_synchronize(split_backend);\n                }")
file(READ "${CMAKE_CURRENT_LIST_DIR}/ordered_expert_sched.inc" ordered_expert)
strata_scheduler_replace("${destination_wait}" "${ordered_expert}\n                if(strata_ordered_weight) {\n                    if(strata_sched_profile.file)++strata_sched_profile.samples[StrataSchedulerProfile::ordered_experts].calls;\n                } else { StrataSchedulerProfile::Timer timer(strata_sched_profile,StrataSchedulerProfile::destination_wait);\n${destination_wait}\n                }")
set(route_get "                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));")
set(route_wait "                        ggml_backend_synchronize(ids_backend);")
strata_scheduler_replace("${route_get}" "                        strata_sched_profile.tokens=ids_tensor->ne[1];\n                        if(strata_reuse_hash_ids(ids_backend,ids_tensor,ids)) {\n                            if(strata_sched_profile.file)++strata_sched_profile.samples[StrataSchedulerProfile::hash_reuses].calls;\n                        } else {\n                        { StrataSchedulerProfile::Timer timer(strata_sched_profile,StrataSchedulerProfile::route_get,ggml_nbytes(ids_tensor));\n${route_get}\n                        }")
strata_scheduler_replace("${route_wait}" "                        { StrataSchedulerProfile::Timer timer(strata_sched_profile,StrataSchedulerProfile::route_wait);\n${route_wait}\n                        }\n                        }")
strata_scheduler_replace("${pipeline_plan}" "                        { StrataSchedulerProfile::Timer timer(strata_sched_profile,StrataSchedulerProfile::plan);\n${pipeline_plan}\n                        }")
strata_scheduler_replace("                    // group consecutive experts and copy them together" "                    StrataSchedulerProfile::Timer copy_timer(strata_sched_profile,StrataSchedulerProfile::copy);\n                    // group consecutive experts and copy them together")
strata_scheduler_replace("            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);" "            enum ggml_status ec;\n            { StrataSchedulerProfile::Timer timer(strata_sched_profile,StrataSchedulerProfile::graph_submit);\n              ec = ggml_backend_graph_compute_async(split_backend, &split->graph); }")
string(PREPEND scheduler "#include \"scheduler_profile.hpp\"\n")
# Keep the route timing on the host thread for the CPU branch's per-job trace.
# route_wait includes preceding GPU work, not just the cost of fetching IDs.
strata_scheduler_replace("                        strata_sched_profile.tokens=ids_tensor->ne[1];"
 "                        strata_sched_profile.tokens=ids_tensor->ne[1];\n                        strata_ds4::CpuRouteProfile::begin(ids_tensor);")
strata_scheduler_replace("                            if(strata_sched_profile.file)++strata_sched_profile.samples[StrataSchedulerProfile::hash_reuses].calls;"
 "                            if(strata_ds4::CpuRouteProfile::enabled())strata_ds4::CpuRouteProfile::current().reused=true;\n                            if(strata_sched_profile.file)++strata_sched_profile.samples[StrataSchedulerProfile::hash_reuses].calls;")
strata_scheduler_replace("${route_get}" "                        strata_ds4::CpuRouteProfile::Timer strata_route_get(false);\n${route_get}")
strata_scheduler_replace("${route_wait}" "                        strata_ds4::CpuRouteProfile::Timer strata_route_wait(true);\n${route_wait}")
string(PREPEND scheduler "#include \"cpu_route_profile.hpp\"\n")
# Launch GPU selection before the existing route read. Hash lookahead already
# has a distinct early transfer plan; leave that path unchanged for now.
strata_scheduler_replace("                        strata_sched_profile.tokens=ids_tensor->ne[1];"
 "                        if(!strata_queue_prepared && ids_backend==split_backend && ids_tensor->op!=GGML_OP_GET_ROWS && strata_pipeline_enabled &&\n                           std::getenv(\"STRATA_DS4_CPU_GPU_QUEUE\") && std::atoi(std::getenv(\"STRATA_DS4_CPU_GPU_QUEUE\"))) {\n                            auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(split_backend));\n#define STRATA_CPU_QUEUE_PREPARE\n#include \"cpu_expert_plan_sched.inc\"\n#undef STRATA_CPU_QUEUE_PREPARE\n                        }\n                        strata_sched_profile.tokens=ids_tensor->ne[1];")
# Insert after the legacy timing-anchor replacement. The new hook also writes
# tokens; applying that replacement to it would publish a CPU job twice when
# fetch declines (for example with the full shadow route recorder enabled).
file(READ "${CMAKE_CURRENT_LIST_DIR}/gpu_fetch_sched.inc" gpu_fetch_sched)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/gpu_fetch_sched.inc")
strata_scheduler_replace("                    if (ids_tensor != prev_ids_tensor) {"
 "${gpu_fetch_sched}\n                    if (ids_tensor != prev_ids_tensor) {")
