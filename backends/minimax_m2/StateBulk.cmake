# Only host state IO changes, in a generated copy of the pinned source.
set(original "${mm27_source}/src/llama-context.cpp")
file(SHA256 "${original}" mm27_state_original_sha256)
if(NOT mm27_state_original_sha256 STREQUAL "2bbf0d5904a4671cb7b5f1309a76b309bc1a06cb7523be9238c6a632b595991c")
  message(FATAL_ERROR "Review MiniMax bulk state IO against changed context")
endif()
file(READ "${original}" content)
string(PREPEND content "#include \"state_bulk.hpp\"\n")
foreach(direction write read)
  if(direction STREQUAL "write")
    set(before [=[        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }]=])
    set(infos winfos)
    set(restore false)
  else()
    set(before [=[        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }]=])
    set(infos rinfos)
    set(restore true)
  endif()
  set(after "        const char * mode = std::getenv(\"STRATA_MM27_STATE_BULK\");\n        minimax_m2_state::flush<${restore}>(${infos}, !(mode && std::strcmp(mode, \"0\") == 0),\n            ggml_backend_tensor_get, ggml_backend_tensor_set, ggml_nbytes);")
  mm27_replace_once(content "${before}" "${after}")
endforeach()
set(generated "${CMAKE_BINARY_DIR}/strata-minimax-m2-context.cpp")
set(previous "")
if(EXISTS "${generated}")
  file(READ "${generated}" previous)
endif()
if(NOT previous STREQUAL content)
  file(WRITE "${generated}" "${content}")
endif()
file(SHA256 "${generated}" mm27_state_generated_sha256)
get_target_property(sources llama SOURCES)
set(matches "${sources}")
list(FILTER matches INCLUDE REGEX "(^|/)llama-context[.]cpp$")
list(LENGTH matches count)
if(NOT count EQUAL 1)
  message(FATAL_ERROR "Expected one llama-context.cpp in llama")
endif()
list(REMOVE_ITEM sources ${matches})
set_property(TARGET llama PROPERTY SOURCES "${sources};${generated}")
set_source_files_properties("${generated}" TARGET_DIRECTORY llama PROPERTIES
  INCLUDE_DIRECTORIES "${mm27_source}/src;${CMAKE_CURRENT_SOURCE_DIR}"
  OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/state_bulk.hpp")
string(APPEND mm27_patch_set ",mm27-bounded-host-state-bulk")
