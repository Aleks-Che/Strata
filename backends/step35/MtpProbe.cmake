# Only the standalone probe links llama-common. Preserve the extracted pin.
set(step_spec_original "${step_source}/common/speculative.cpp")
file(SHA256 "${step_spec_original}" step_spec_hash)
if(NOT step_spec_hash STREQUAL "c5f070dfc9d3d1af31f92567900d6bb06a8d35fe5fcef6df7cd416be1758dbc2")
  message(FATAL_ERROR "Review Step MTP catch-up patch for the changed speculative driver")
endif()
file(READ "${step_spec_original}" content)
# A fresh driver has a fixed draft depth for the entire request. Deeper heads
# are never sampled and need no teacher-forced KV catch-up. This also permits
# their weights to remain mapped in RAM in the optional active-heads placement.
step_replace_once(content
  "for (int head = 0; head < n_mtp_layers; ++head) {"
  "for (int head = 0; head < std::min(n_mtp_layers, params.n_max); ++head) {")
set(generated "${CMAKE_BINARY_DIR}/strata-step35-speculative.cpp")
set(previous "")
if(EXISTS "${generated}")
  file(READ "${generated}" previous)
endif()
if(NOT previous STREQUAL content)
  file(WRITE "${generated}" "${content}")
endif()
get_target_property(sources llama-common SOURCES)
set(matches "${sources}")
list(FILTER matches INCLUDE REGEX "(^|/)speculative.cpp$")
list(LENGTH matches count)
if(NOT count EQUAL 1)
  message(FATAL_ERROR "Expected one speculative.cpp in llama-common")
endif()
list(REMOVE_ITEM sources ${matches})
set_property(TARGET llama-common PROPERTY SOURCES "${sources};${generated}")
set_source_files_properties("${generated}" TARGET_DIRECTORY llama-common PROPERTIES
  INCLUDE_DIRECTORIES "${step_source}/common")
