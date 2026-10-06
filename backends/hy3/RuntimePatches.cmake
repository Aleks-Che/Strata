# Private Hy3 scheduler, demand mapping and native-file registration.
function(hy3_replace_once variable before after)
  string(FIND "${${variable}}" "${before}" pos)
  if(pos EQUAL -1)
    message(FATAL_ERROR "Missing Hy3 runtime patch anchor: ${before}")
  endif()
  string(REPLACE "${before}" "" without "${${variable}}")
  # Hash guards pin the entire file; anchors are additionally checked below.
  string(LENGTH "${${variable}}" old_length)
  string(LENGTH "${without}" new_length)
  string(LENGTH "${before}" anchor_length)
  math(EXPR removed "${old_length}-${new_length}")
  if(NOT removed EQUAL anchor_length)
    message(FATAL_ERROR "Expected one Hy3 runtime anchor: ${before}")
  endif()
  string(REPLACE "${before}" "${after}" result "${${variable}}")
  set(${variable} "${result}" PARENT_SCOPE)
endfunction()
function(hy3_runtime_source target relative expected output)
  set(original "${hy3_source}/${relative}")
  file(SHA256 "${original}" hash)
  if(NOT hash STREQUAL expected)
    message(FATAL_ERROR "Review Hy3 runtime patch for changed ${relative}")
  endif()
  file(READ "${original}" content)
  if(target STREQUAL "ggml-base")
    set(anchor "static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {")
    hy3_replace_once(content "${anchor}" "#include \"sync_runtime.inc\"\n${anchor}\n#include \"gpu_only_audit.inc\"\n    Hy3PlanScope hy3_scope;")
    hy3_replace_once(content "                        prev_ids_tensor = ids_tensor;"
      "#include \"pipeline_sched.inc\"\n                        prev_ids_tensor = ids_tensor;")
    hy3_replace_once(content "                    // group consecutive experts and copy them together"
      "                    if (hy3_sync_mode == 2 && hy3_pipeline && hy3_pipeline_batch) {\n                        if (!hy3_pipeline_copy_tensor(split_backend, input_cpy, input)) return GGML_STATUS_FAILED;\n                        continue;\n                    }\n\n                    // group consecutive experts and copy them together")
    hy3_replace_once(content "        prev_backend_id = split_backend_id;\n    }\n\n    return GGML_STATUS_SUCCESS;"
      "        prev_backend_id = split_backend_id;\n    }\n\n    if (!hy3_scope.finish()) return GGML_STATUS_FAILED;\n    return GGML_STATUS_SUCCESS;")
    hy3_replace_once(content "ggml_backend_graph_compute_async(split_backend, &split->graph)" "hy3_graph_compute(split_backend, &split->graph)")
    hy3_replace_once(content "ggml_backend_graph_compute_async(split_backend, &gv)" "hy3_graph_compute(split_backend, &gv)")
    hy3_replace_once(content "        struct ggml_backend_sched_split * split = &splits[split_id];"
      "        if (hy3_sync_mode && hy3_cancelled()) return GGML_STATUS_ABORTED;\n        struct ggml_backend_sched_split * split = &splits[split_id];")
    set(anchor "                        ggml_backend_tensor_set_async(split_backend,")
    hy3_replace_once(content "${anchor}" "                        if (hy3_sync_mode == 2) {\n                            return hy3_copy_range(split_backend, input_cpy, input, expert_offset, expert_size_copy + padding_end);\n                        }\n                        if (hy3_sync_mode == 1) {\n                            ++hy3_sync_stats.ranges;\n                            hy3_sync_stats.h2d_bytes += expert_size_copy + padding_end;\n                        }\n${anchor}")
    hy3_replace_once(content "                            expert_size_copy + padding_end);\n                    };"
      "                            expert_size_copy + padding_end);\n                        return true;\n                    };")
    hy3_replace_once(content "                        copy_experts(first_id, last_id);"
      "                        if (!copy_experts(first_id, last_id)) return GGML_STATUS_FAILED;")
    hy3_replace_once(content "                    copy_experts(first_id, last_id);"
      "                    if (!copy_experts(first_id, last_id)) return GGML_STATUS_FAILED;")
    set(anchor "                    // try async copy, but if not possible,")
    hy3_replace_once(content "${anchor}" "                    if (hy3_sync_mode && ggml_backend_buffer_is_host(input->buffer) &&\n                        ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && hy3_expert_weight(input)) {\n                        ++hy3_sync_stats.rejected_full_copies;\n                        GGML_LOG_ERROR(\"STRATA_HY3 refusing full expert copy: %s\\n\",input->name);\n                        return GGML_STATUS_FAILED;\n                    }\n${anchor}")
  elseif(relative STREQUAL "src/llama-mmap.cpp")
    string(PREPEND content "#include \"../common/expert_file.hpp\"\n")
    set(anchor "        if (prefetch > 0) {\n#if _WIN32_WINNT >= 0x602")
    hy3_replace_once(content "${anchor}" "        try {\n            strata_expert_file::add(addr, size, file->file_id());\n        } catch (...) {\n            UnmapViewOfFile(addr);\n            CloseHandle(hMapping);\n            throw;\n        }\n${anchor}")
    hy3_replace_once(content "            if (addr) {\n                if (!UnmapViewOfFile(addr)) {"
      "            if (addr) {\n                strata_expert_file::remove(addr);\n                if (!UnmapViewOfFile(addr)) {")
  else()
    hy3_replace_once(content "const size_t prefetch_size = prefetch && use_mmap ? -1 : 0;"
      "const size_t prefetch_size = arch_name == \"hy_v3\" ? 0 : (prefetch && use_mmap ? -1 : 0); // demand-page Hy3 experts")
  endif()
  set(generated "${CMAKE_BINARY_DIR}/${output}")
  set(previous "")
  if(EXISTS "${generated}")
    file(READ "${generated}" previous)
  endif()
  if(NOT previous STREQUAL content)
    file(WRITE "${generated}" "${content}")
  endif()
  file(SHA256 "${generated}" generated_hash)
  set_property(GLOBAL APPEND PROPERTY HY3_RUNTIME_HASHES "${output}:${generated_hash}")
  get_target_property(sources ${target} SOURCES)
  get_filename_component(name "${relative}" NAME)
  set(matches "${sources}")
  list(FILTER matches INCLUDE REGEX "(^|/)${name}$")
  list(LENGTH matches count)
  if(NOT count EQUAL 1)
    message(FATAL_ERROR "Expected one ${name} in ${target}")
  endif()
  list(REMOVE_ITEM sources ${matches})
  set_property(TARGET ${target} PROPERTY SOURCES "${sources};${generated}")
  get_filename_component(directory "${original}" DIRECTORY)
  set_source_files_properties("${generated}" TARGET_DIRECTORY ${target} PROPERTIES
    INCLUDE_DIRECTORIES "${directory};${CMAKE_CURRENT_SOURCE_DIR}"
    OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/sync_runtime.h;${CMAKE_CURRENT_SOURCE_DIR}/sync_test.h;${CMAKE_CURRENT_SOURCE_DIR}/sync_runtime.inc;${CMAKE_CURRENT_SOURCE_DIR}/gpu_only_audit.inc;${CMAKE_CURRENT_SOURCE_DIR}/../common/expert_file.hpp;${CMAKE_CURRENT_SOURCE_DIR}/../common/device_memory.hpp;${CMAKE_CURRENT_SOURCE_DIR}/../glm5next/host_pages.hpp")
endfunction()
hy3_runtime_source(ggml-base ggml/src/ggml-backend.cpp
  a39c4fe81b043c7e8616ebe57afb75d727c692fe3b26c3e9bc2ddde3c6991041 strata-hy3-backend.cpp)
hy3_runtime_source(llama src/llama-model-loader.cpp
  5ef07476310d4678df18a61a6ec0a1ebcbb58c7534ac3c624b9fe0f315a4ab01 strata-hy3-loader.cpp)
hy3_runtime_source(llama src/llama-mmap.cpp
  3ca6869dfccbdbbafad0802e1a3d7db52174347d36174a982c1a662d7034b9c6 strata-hy3-mmap.cpp)
get_property(hy3_runtime_hashes GLOBAL PROPERTY HY3_RUNTIME_HASHES)
set(cache_dependencies "${CMAKE_CURRENT_SOURCE_DIR}/cache_runtime.inc"
  "${CMAKE_CURRENT_SOURCE_DIR}/pipeline_runtime.inc"
  "${CMAKE_CURRENT_SOURCE_DIR}/pipeline_sched.inc"
  "${CMAKE_CURRENT_SOURCE_DIR}/../common/expert_pipeline.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../common/expert_slice.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../step35/gpu_trace.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../step35/expert_cache.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../common/vram_policy.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../common/expert_frequency.hpp")
set_property(SOURCE "${CMAKE_BINARY_DIR}/strata-hy3-backend.cpp" TARGET_DIRECTORY ggml-base APPEND
  PROPERTY OBJECT_DEPENDS "${cache_dependencies}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${cache_dependencies}")
foreach(dependency IN LISTS cache_dependencies)
  file(SHA256 "${dependency}" dependency_hash)
  get_filename_component(dependency_name "${dependency}" NAME)
  list(APPEND hy3_runtime_hashes "${dependency_name}:${dependency_hash}")
endforeach()
string(APPEND hy3_patch_set ",hy3-sync-selected-file-copy-gpu-audit-demand-mmap,hy3-bounded-matrix-cache,hy3-bounded-pipeline,hy3-tensor-batch-copy")
target_link_libraries(ggml-base PRIVATE CUDA::cudart_static)
target_include_directories(ggml-base PRIVATE "${hy3_source}/vendor")
target_compile_features(ggml-base PRIVATE cxx_std_17)
