# Private generated copies; upstream source and other model backends stay intact.
function(glm_runtime_source target relative expected_hash output_name)
  set(original "${glm_source}/${relative}")
  file(SHA256 "${original}" actual_hash)
  if(NOT actual_hash STREQUAL expected_hash)
    message(FATAL_ERROR "Review GLM runtime patch for changed ${relative}")
  endif()
  file(READ "${original}" source_text)
  if(target STREQUAL "ggml-base")
    set(wait_before "                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }")
    string(FIND "${source_text}" "${wait_before}" wait_found)
    if(wait_found EQUAL -1)
      message(FATAL_ERROR "GLM scheduler scratch-wait boundary not found")
    endif()
    # Only the selected-expert hook owns the replacement dependency chain.
    # All other input copies, including user inputs, retain upstream waits.
    string(REPLACE "${wait_before}" "                const bool strata_event_input = strata_glm_sync_active && !strata_glm_candidate_copy &&
                    strata_glm_event_copy && strata_glm_copy_callback && split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_is_host(input->buffer) &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    split->graph.nodes[0]->op == GGML_OP_MUL_MAT_ID && split->graph.nodes[0]->src[0] == input_cpy;
                if (!strata_event_input) {
${wait_before}
                }" source_text "${source_text}")
    string(REPLACE "ggml_backend_graph_compute_async(split_backend," "strata_glm_graph_compute(split_backend," source_text "${source_text}")
    # The implementation needs backend types and APIs, so put it after includes.
    string(REPLACE "static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {"
      "#include \"sync_runtime.inc\"\nstatic enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {\n#include \"gpu_only_audit.inc\"\n    strata_glm_plan_scope strata_glm_scope;"
      source_text "${source_text}")
    string(REPLACE "                        ggml_backend_tensor_set_async(split_backend,"
      "                        if (strata_glm_sync_active && !strata_glm_candidate_copy) {\n                            strata_glm_copy_range(split_backend, input_cpy, input, expert_offset, expert_size_copy + padding_end, last_id-first_id+1);\n                            return;\n                        }\n                        ggml_backend_tensor_set_async(split_backend,"
      source_text "${source_text}")
    string(REPLACE "                        prev_ids_tensor = ids_tensor;"
      "#include \"pipeline_sched.inc\"\n                        prev_ids_tensor = ids_tensor;" source_text "${source_text}")
    string(REPLACE "        prev_backend_id = split_backend_id;\n    }\n\n    return GGML_STATUS_SUCCESS;"
      "        prev_backend_id = split_backend_id;\n    }\n\n    strata_glm_scope.finish();\n    return GGML_STATUS_SUCCESS;" source_text "${source_text}")
    string(REPLACE "                    // try async copy, but if not possible,"
      "                    if (strata_glm_sync_active && ggml_backend_buffer_is_host(input->buffer) &&\n                        ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&\n                        strstr(input->name, \"_exps.weight\")) {\n                        GGML_LOG_ERROR(\"STRATA_GLM refusing full expert tensor copy: %s\\n\", input->name);\n                        return GGML_STATUS_FAILED;\n                    }\n                    // try async copy, but if not possible,"
      source_text "${source_text}")
  elseif(relative STREQUAL "src/llama-mmap.cpp")
    # Register the exact offset-zero mapping/descriptor, including split GGUFs.
    # A reopened handle belongs to each queued read; the model owns the view.
    string(PREPEND source_text "#include \"../common/expert_file.hpp\"\n")
    set(map_anchor "        if (prefetch > 0) {\n#if _WIN32_WINNT >= 0x602")
    string(FIND "${source_text}" "${map_anchor}" map_found)
    if(map_found EQUAL -1)
      message(FATAL_ERROR "GLM Windows mapping registration boundary not found")
    endif()
    string(REPLACE "${map_anchor}" "        try {
            strata_expert_file::add(addr, size, file->file_id());
        } catch (...) {
            UnmapViewOfFile(addr);
            CloseHandle(hMapping);
            throw;
        }

${map_anchor}" source_text "${source_text}")
    string(REPLACE "            if (addr) {\n                if (!UnmapViewOfFile(addr)) {"
      "            if (addr) {\n                strata_expert_file::remove(addr);\n                if (!UnmapViewOfFile(addr)) {" source_text "${source_text}")
  else()
    string(REPLACE "const size_t prefetch_size = prefetch && use_mmap ? -1 : 0;"
      "const size_t prefetch_size = 0; // Strata: fault expert ranges only on demand."
      source_text "${source_text}")
  endif()
  set(generated "${CMAKE_BINARY_DIR}/${output_name}")
  set(old "")
  if(EXISTS "${generated}")
    file(READ "${generated}" old)
  endif()
  if(NOT old STREQUAL source_text)
    file(WRITE "${generated}" "${source_text}")
  endif()
  get_target_property(sources ${target} SOURCES)
  get_filename_component(basename "${relative}" NAME)
  set(matches "${sources}")
  list(FILTER matches INCLUDE REGEX "(^|/)${basename}$")
  list(LENGTH matches count)
  if(NOT count EQUAL 1)
    message(FATAL_ERROR "Expected one ${basename} in ${target}")
  endif()
  list(REMOVE_ITEM sources ${matches})
  set_property(TARGET ${target} PROPERTY SOURCES "${sources};${generated}")
  get_filename_component(original_dir "${original}" DIRECTORY)
  set_source_files_properties("${generated}" TARGET_DIRECTORY ${target} PROPERTIES
    INCLUDE_DIRECTORIES "${original_dir};${CMAKE_CURRENT_SOURCE_DIR}"
    OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/sync_runtime.h;${CMAKE_CURRENT_SOURCE_DIR}/sync_runtime.inc;${CMAKE_CURRENT_SOURCE_DIR}/gpu_only_audit.inc;${CMAKE_CURRENT_SOURCE_DIR}/pipeline_sched.inc;${CMAKE_CURRENT_SOURCE_DIR}/../common/expert_file.hpp")
endfunction()
glm_runtime_source(ggml-base ggml/src/ggml-backend.cpp
  a39c4fe81b043c7e8616ebe57afb75d727c692fe3b26c3e9bc2ddde3c6991041 strata-glm-backend.cpp)
glm_runtime_source(llama src/llama-model-loader.cpp
  5ef07476310d4678df18a61a6ec0a1ebcbb58c7534ac3c624b9fe0f315a4ab01 strata-glm-loader.cpp)
glm_runtime_source(llama src/llama-mmap.cpp
  3ca6869dfccbdbbafad0802e1a3d7db52174347d36174a982c1a662d7034b9c6 strata-glm-mmap.cpp)
string(APPEND glm_candidate_patches ",sync-selected-experts-16MiB-pinned,gpu-only-precompute-audit,mmap-no-prefetch,runtime-cache,router-lookahead-pipeline,native-mtp-rollback,optional-event-fenced-expert-copy,optional-pipeline-staging-tuning,optional-main-cache-aging,optional-pool-reclaim-before-budget,optional-cache-allocator,bounded-ram-warmup,optional-packed-expert-cache,optional-learned-expert-warmup,optional-vram-aware-host-pages")
string(APPEND glm_candidate_patches ",optional-native-expert-reads,optional-idle-slab-compaction")
