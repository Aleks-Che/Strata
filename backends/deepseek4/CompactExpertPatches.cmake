file(READ "${CMAKE_CURRENT_LIST_DIR}/compact_expert_sched.inc" compact_sched)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/compact_expert_sched.inc")
set(compact_anchor "            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);")
strata_scheduler_replace("${compact_anchor}" "${compact_anchor}\n${compact_sched}")
strata_scheduler_replace("        assert(size >= ggml_nbytes(tensor));"
  "        assert(size >= ggml_nbytes(tensor) || size == strata_compact_bytes(tensor));")
string(PREPEND scheduler "#include \"expert_compact.hpp\"\n")
