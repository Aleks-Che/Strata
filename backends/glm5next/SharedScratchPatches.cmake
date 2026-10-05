# Private allocator copy; vbuffers gain shared ownership only through our
# explicit setup-time API. A later reserve that grows detaches safely.
set(glm_alloc_original "${glm_source}/ggml/src/ggml-alloc.c")
file(SHA256 "${glm_alloc_original}" glm_alloc_hash)
if(NOT glm_alloc_hash STREQUAL "213b48d6ef6d6ffef55c438db837116105de72cdae3dd152673035e61508d384")
  message(FATAL_ERROR "Review GLM shared scratch allocator patch for changed source")
endif()
file(READ "${glm_alloc_original}" glm_alloc_source)
string(PREPEND glm_alloc_source "#include \"shared_scratch.h\"\n")
string(REPLACE "struct vbuffer {" "struct vbuffer {\n    size_t strata_refs;" glm_alloc_source "${glm_alloc_source}")
string(REPLACE "    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {\n        ggml_backend_buffer_free"
  "    GGML_ASSERT(buf->strata_refs > 0);\n    if (--buf->strata_refs != 0) return;\n    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {\n        ggml_backend_buffer_free"
  glm_alloc_source "${glm_alloc_source}")
string(REPLACE "    for (int n = 0; n < talloc->n_chunks; n++) {"
  "    buf->strata_refs = 1;\n    for (int n = 0; n < talloc->n_chunks; n++) {" glm_alloc_source "${glm_alloc_source}")
string(REPLACE "ggml_gallocr_t ggml_gallocr_new_n("
  "#include \"shared_scratch_alloc.inc\"\n\nggml_gallocr_t ggml_gallocr_new_n(" glm_alloc_source "${glm_alloc_source}")
set(glm_alloc_generated "${CMAKE_BINARY_DIR}/strata-glm-alloc.c")
set(glm_alloc_old "")
if(EXISTS "${glm_alloc_generated}")
  file(READ "${glm_alloc_generated}" glm_alloc_old)
endif()
if(NOT glm_alloc_old STREQUAL glm_alloc_source)
  file(WRITE "${glm_alloc_generated}" "${glm_alloc_source}")
endif()
get_target_property(glm_base_sources ggml-base SOURCES)
set(glm_alloc_matches "${glm_base_sources}")
list(FILTER glm_alloc_matches INCLUDE REGEX "(^|/)ggml-alloc\\.c$")
list(LENGTH glm_alloc_matches glm_alloc_count)
if(NOT glm_alloc_count EQUAL 1)
  message(FATAL_ERROR "Expected one ggml-alloc.c")
endif()
list(REMOVE_ITEM glm_base_sources ${glm_alloc_matches})
set_property(TARGET ggml-base PROPERTY SOURCES "${glm_base_sources};${glm_alloc_generated}")
set_source_files_properties("${glm_alloc_generated}" TARGET_DIRECTORY ggml-base PROPERTIES
  INCLUDE_DIRECTORIES "${glm_source}/ggml/src;${CMAKE_CURRENT_SOURCE_DIR}"
  OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shared_scratch.h;${CMAKE_CURRENT_SOURCE_DIR}/shared_scratch_alloc.inc")
string(APPEND glm_candidate_patches ",optional-shared-mtp-scratch")

# NextN feature extraction changes liveness of the target's hidden rows. Reserve
# with those final output flags before sharing, otherwise the first real graph
# may grow its buffer and immediately detach from the shared allocation.
set(glm_context_original "${glm_source}/src/llama-context.cpp")
file(SHA256 "${glm_context_original}" glm_context_hash)
if(NOT glm_context_hash STREQUAL "2bbf0d5904a4671cb7b5f1309a76b309bc1a06cb7523be9238c6a632b595991c")
  message(FATAL_ERROR "Review GLM shared scratch context reservation patch")
endif()
file(READ "${glm_context_original}" glm_context_source)
string(REPLACE "    cparams.embeddings_nextn        = value;"
  "    const char * strata_shared = getenv(\"STRATA_GLM_MTP_SHARED_SCRATCH\");\n    if (strata_shared && strata_shared[0]=='1' && strata_shared[1]=='\\0' &&\n        (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked)) {\n        sched_need_reserve = true;\n    }\n    cparams.embeddings_nextn        = value;"
  glm_context_source "${glm_context_source}")
set(glm_context_generated "${CMAKE_BINARY_DIR}/strata-glm-context.cpp")
set(glm_context_old "")
if(EXISTS "${glm_context_generated}")
  file(READ "${glm_context_generated}" glm_context_old)
endif()
if(NOT glm_context_old STREQUAL glm_context_source)
  file(WRITE "${glm_context_generated}" "${glm_context_source}")
endif()
get_target_property(glm_llama_sources llama SOURCES)
set(glm_context_matches "${glm_llama_sources}")
list(FILTER glm_context_matches INCLUDE REGEX "(^|/)llama-context\\.cpp$")
list(LENGTH glm_context_matches glm_context_count)
if(NOT glm_context_count EQUAL 1)
  message(FATAL_ERROR "Expected one llama-context.cpp")
endif()
list(REMOVE_ITEM glm_llama_sources ${glm_context_matches})
set_property(TARGET llama PROPERTY SOURCES "${glm_llama_sources};${glm_context_generated}")
set_source_files_properties("${glm_context_generated}" TARGET_DIRECTORY llama PROPERTIES
  INCLUDE_DIRECTORIES "${glm_source}/src")
