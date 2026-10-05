# Private allocator copy; vbuffers gain shared ownership only through our
# explicit setup-time API. A later reserve that grows detaches safely.
set(ds4_alloc_original "${STRATA_LLAMA_DIR}/ggml/src/ggml-alloc.c")
file(SHA256 "${ds4_alloc_original}" ds4_alloc_hash)
if(NOT ds4_alloc_hash STREQUAL "213b48d6ef6d6ffef55c438db837116105de72cdae3dd152673035e61508d384")
  message(FATAL_ERROR "Review DeepSeek shared scratch allocator patch for changed source")
endif()
file(READ "${ds4_alloc_original}" ds4_alloc_source)
string(PREPEND ds4_alloc_source "#include \"shared_scratch.h\"\n")
string(REPLACE "struct vbuffer {" "struct vbuffer {\n    size_t strata_refs;" ds4_alloc_source "${ds4_alloc_source}")
string(REPLACE "    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {\n        ggml_backend_buffer_free"
  "    GGML_ASSERT(buf->strata_refs > 0);\n    if (--buf->strata_refs != 0) return;\n    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {\n        ggml_backend_buffer_free"
  ds4_alloc_source "${ds4_alloc_source}")
string(REPLACE "    for (int n = 0; n < talloc->n_chunks; n++) {"
  "    buf->strata_refs = 1;\n    for (int n = 0; n < talloc->n_chunks; n++) {" ds4_alloc_source "${ds4_alloc_source}")
string(REPLACE "ggml_gallocr_t ggml_gallocr_new_n("
  "#include \"shared_scratch_alloc.inc\"\n\nggml_gallocr_t ggml_gallocr_new_n(" ds4_alloc_source "${ds4_alloc_source}")
set(ds4_alloc_generated "${CMAKE_BINARY_DIR}/strata-ds4-alloc.c")
set(ds4_alloc_old "")
if(EXISTS "${ds4_alloc_generated}")
  file(READ "${ds4_alloc_generated}" ds4_alloc_old)
endif()
if(NOT ds4_alloc_old STREQUAL ds4_alloc_source)
  file(WRITE "${ds4_alloc_generated}" "${ds4_alloc_source}")
endif()
get_target_property(ds4_base_sources ggml-base SOURCES)
set(ds4_alloc_matches "${ds4_base_sources}")
list(FILTER ds4_alloc_matches INCLUDE REGEX "(^|/)ggml-alloc\\.c$")
list(LENGTH ds4_alloc_matches ds4_alloc_count)
if(NOT ds4_alloc_count EQUAL 1)
  message(FATAL_ERROR "Expected one ggml-alloc.c")
endif()
list(REMOVE_ITEM ds4_base_sources ${ds4_alloc_matches})
set_property(TARGET ggml-base PROPERTY SOURCES "${ds4_base_sources};${ds4_alloc_generated}")
set_source_files_properties("${ds4_alloc_generated}" TARGET_DIRECTORY ggml-base PROPERTIES
  INCLUDE_DIRECTORIES "${STRATA_LLAMA_DIR}/ggml/src;${CMAKE_CURRENT_SOURCE_DIR}"
  OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shared_scratch.h;${CMAKE_CURRENT_SOURCE_DIR}/shared_scratch_alloc.inc")


