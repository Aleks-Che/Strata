# Private allocator copy; vbuffers gain shared ownership only through our
# explicit setup-time API. A later reserve that grows detaches safely.
set(hy3_alloc_original "${hy3_source}/ggml/src/ggml-alloc.c")
file(SHA256 "${hy3_alloc_original}" hy3_alloc_hash)
if(NOT hy3_alloc_hash STREQUAL "213b48d6ef6d6ffef55c438db837116105de72cdae3dd152673035e61508d384")
  message(FATAL_ERROR "Review Hy3 shared scratch allocator patch for changed source")
endif()
file(READ "${hy3_alloc_original}" hy3_alloc_source)
string(PREPEND hy3_alloc_source "#include \"shared_scratch.h\"\n")
string(REPLACE "struct vbuffer {" "struct vbuffer {\n    size_t strata_refs;" hy3_alloc_source "${hy3_alloc_source}")
string(REPLACE "    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {\n        ggml_backend_buffer_free"
  "    GGML_ASSERT(buf->strata_refs > 0);\n    if (--buf->strata_refs != 0) return;\n    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {\n        ggml_backend_buffer_free"
  hy3_alloc_source "${hy3_alloc_source}")
string(REPLACE "    for (int n = 0; n < talloc->n_chunks; n++) {"
  "    buf->strata_refs = 1;\n    for (int n = 0; n < talloc->n_chunks; n++) {" hy3_alloc_source "${hy3_alloc_source}")
string(REPLACE "ggml_gallocr_t ggml_gallocr_new_n("
  "#include \"shared_scratch_alloc.inc\"\n\nggml_gallocr_t ggml_gallocr_new_n(" hy3_alloc_source "${hy3_alloc_source}")
set(hy3_alloc_generated "${CMAKE_BINARY_DIR}/strata-hy3-alloc.c")
set(hy3_alloc_old "")
if(EXISTS "${hy3_alloc_generated}")
  file(READ "${hy3_alloc_generated}" hy3_alloc_old)
endif()
if(NOT hy3_alloc_old STREQUAL hy3_alloc_source)
  file(WRITE "${hy3_alloc_generated}" "${hy3_alloc_source}")
endif()
get_target_property(hy3_base_sources ggml-base SOURCES)
set(hy3_alloc_matches "${hy3_base_sources}")
list(FILTER hy3_alloc_matches INCLUDE REGEX "(^|/)ggml-alloc\\.c$")
list(LENGTH hy3_alloc_matches hy3_alloc_count)
if(NOT hy3_alloc_count EQUAL 1)
  message(FATAL_ERROR "Expected one ggml-alloc.c")
endif()
list(REMOVE_ITEM hy3_base_sources ${hy3_alloc_matches})
set_property(TARGET ggml-base PROPERTY SOURCES "${hy3_base_sources};${hy3_alloc_generated}")
set_source_files_properties("${hy3_alloc_generated}" TARGET_DIRECTORY ggml-base PROPERTIES
  INCLUDE_DIRECTORIES "${hy3_source}/ggml/src;${CMAKE_CURRENT_SOURCE_DIR}"
  OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shared_scratch.h;${CMAKE_CURRENT_SOURCE_DIR}/shared_scratch_alloc.inc")


