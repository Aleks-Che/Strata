# A separate template oracle: keep the original raw target and dependency intact.
set(json_value_source "${mm27_source}/common/jinja/value.cpp")
file(SHA256 "${json_value_source}" mm27_template_json_original_sha256)
if(NOT mm27_template_json_original_sha256 STREQUAL "cf2f6f137c3eadb0a61a88a7ca7fd0be161f6df38a5b661b3e89e1006dca48bd")
  message(FATAL_ERROR "Unreviewed native Jinja JSON serializer")
endif()
file(READ "${json_value_source}" json_value_text)
set(json_float_before "oss << val->as_float();")
set(json_float_after "oss << common_json(val->as_float()).dump();")
string(FIND "${json_value_text}" "${json_float_before}" json_float_at)
if(json_float_at LESS 0)
  message(FATAL_ERROR "Native Jinja float serializer patch target missing")
endif()
string(REPLACE "${json_float_before}" "${json_float_after}" json_value_text "${json_value_text}")
foreach(header runtime value)
  string(REPLACE "#include \"${header}.h\"" "#include \"jinja/${header}.h\"" json_value_text "${json_value_text}")
endforeach()
set(json_value_generated "${CMAKE_BINARY_DIR}/minimax-template-json-value.cpp")
file(CONFIGURE OUTPUT "${json_value_generated}" CONTENT "${json_value_text}" @ONLY)
file(SHA256 "${json_value_generated}" mm27_template_json_generated_sha256)
file(SHA256 "${CMAKE_CURRENT_LIST_FILE}" mm27_template_json_patch_sha256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${json_value_source}" "${CMAKE_CURRENT_LIST_FILE}")
add_executable(strata-minimax-m2-template-json template_oracle.cpp "${mm27_source}/common/json.cpp"
  "${mm27_source}/common/unicode.cpp" "${mm27_source}/common/jinja/lexer.cpp"
  "${mm27_source}/common/jinja/parser.cpp" "${mm27_source}/common/jinja/runtime.cpp"
  "${json_value_generated}" "${mm27_source}/common/jinja/string.cpp" "${mm27_source}/common/jinja/caps.cpp")
target_include_directories(strata-minimax-m2-template-json PRIVATE
  "${mm27_source}/common" "${mm27_source}/vendor")
target_link_libraries(strata-minimax-m2-template-json PRIVATE ggml-base)
target_compile_features(strata-minimax-m2-template-json PRIVATE cxx_std_17)
target_compile_definitions(strata-minimax-m2-template-json PRIVATE STRATA_MM27_TEMPLATE_JSON=1
  STRATA_MM27_SOURCE_SHA="${STRATA_MM27_SOURCE_SHA}" STRATA_MM27_ARCHIVE_SHA256="${STRATA_MM27_ARCHIVE_SHA256}"
  STRATA_MM27_PATCH_SET="${mm27_patch_set}")
if(MSVC)
  target_compile_options(strata-minimax-m2-template-json PRIVATE /utf-8 /bigobj)
endif()
set_target_properties(strata-minimax-m2-template-json PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME minimax_m2_template_json_version COMMAND strata-minimax-m2-template-json --version)
set_tests_properties(minimax_m2_template_json_version PROPERTIES TIMEOUT 15)
