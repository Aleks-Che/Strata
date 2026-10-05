# Preserve the pinned source archive; compile a generated GLM graph instead.
set(glm_mtp_original "${glm_source}/src/models/glm5next.cpp")
file(SHA256 "${glm_mtp_original}" glm_mtp_hash)
if(NOT glm_mtp_hash STREQUAL "3ccdb68006c92060270b90561a1fc9ff26bfa41cf0f24987b9eee72f513342a5")
  message(FATAL_ERROR "Review GLM MTP cache-only patch for changed model source")
endif()
file(READ "${glm_mtp_original}" glm_mtp_source)
# A null query is used only by the cache-only NextN graph. Preserve pooling
# writes before skipping selection; the normal target/draft paths keep qr.
set(glm_mtp_anchor "    ggml_tensor * pooled_rd = ggml_view_3d(ctx0, kbuf, d_idx, n_kv, n_stream,")
string(REPLACE "${glm_mtp_anchor}" "    if (qr == nullptr) { return nullptr; } // Strata cache-only MTP\n\n${glm_mtp_anchor}"
  glm_mtp_source "${glm_mtp_source}")
# Change only the NextN constructor, not the target constructor's output IDs.
string(FIND "${glm_mtp_source}" "llama_model_glm5next::graph_mtp::graph_mtp" glm_mtp_start)
if(glm_mtp_start EQUAL -1)
  message(FATAL_ERROR "GLM MTP constructor not found")
endif()
string(SUBSTRING "${glm_mtp_source}" 0 ${glm_mtp_start} glm_mtp_prefix)
string(SUBSTRING "${glm_mtp_source}" ${glm_mtp_start} -1 glm_mtp_body)
string(REPLACE "    ggml_tensor * inp_out_ids = build_inp_out_ids();"
  "    const bool cache_only = n_outputs == 0 && !cparams.embeddings &&\n        !(cparams.embeddings_nextn && !cparams.embeddings_nextn_masked);\n    ggml_tensor * inp_out_ids = cache_only ? nullptr : build_inp_out_ids();"
  glm_mtp_body "${glm_mtp_body}")
string(REPLACE "    cb(cur, \"mtp_attn_norm\", il);"
  "    cb(cur, \"mtp_attn_norm\", il);\n#include \"mtp_cache_only.inc\""
  glm_mtp_body "${glm_mtp_body}")
set(glm_mtp_source "${glm_mtp_prefix}${glm_mtp_body}")
set(glm_mtp_generated "${CMAKE_BINARY_DIR}/strata-glm-model.cpp")
set(glm_mtp_old "")
if(EXISTS "${glm_mtp_generated}")
  file(READ "${glm_mtp_generated}" glm_mtp_old)
endif()
if(NOT glm_mtp_old STREQUAL glm_mtp_source)
  file(WRITE "${glm_mtp_generated}" "${glm_mtp_source}")
endif()
get_target_property(glm_mtp_sources llama SOURCES)
set(glm_mtp_matches "${glm_mtp_sources}")
list(FILTER glm_mtp_matches INCLUDE REGEX "(^|/)glm5next\\.cpp$")
list(LENGTH glm_mtp_matches glm_mtp_count)
if(NOT glm_mtp_count EQUAL 1)
  message(FATAL_ERROR "Expected one glm5next.cpp in llama")
endif()
list(REMOVE_ITEM glm_mtp_sources ${glm_mtp_matches})
set_property(TARGET llama PROPERTY SOURCES "${glm_mtp_sources};${glm_mtp_generated}")
set_source_files_properties("${glm_mtp_generated}" TARGET_DIRECTORY llama PROPERTIES
  INCLUDE_DIRECTORIES "${glm_source}/src/models;${CMAKE_CURRENT_SOURCE_DIR}"
  OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mtp_cache_only.inc")
string(APPEND glm_candidate_patches ",mtp-cache-only-catch-up")
