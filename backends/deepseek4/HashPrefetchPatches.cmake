# Exact hash lookahead changes I/O scheduling only, never the model graph.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/hash_prefetch_sched.inc")
file(READ "${CMAKE_CURRENT_LIST_DIR}/hash_prefetch_sched.inc" hash_prefetch)
strata_scheduler_replace("    int prev_backend_id = -1;"
  "    int prev_backend_id = -1;\n${hash_prefetch}")
# A new plan may retire direct-cache leases. Only do this after the final
# consumer graph has been submitted and its CUDA reader fences are recorded.
strata_scheduler_replace("        // record the event of this split"
  "        for(size_t h=0;h+1<strata_hash_groups.size();++h)\n            if(split_id==strata_hash_groups[h].last_split)strata_start_hash_group(h+1);\n\n        // record the event of this split")
