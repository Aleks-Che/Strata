# Narrow output adapter for the reviewed MiniMax DFlash sidecars. Keep the
# borrowed weight unchanged; compact every logit row before context output copy.
set(mm27_dflash_original "${mm27_source}/src/models/dflash.cpp")
file(SHA256 "${mm27_dflash_original}" mm27_dflash_original_sha256)
if(NOT mm27_dflash_original_sha256 STREQUAL "4a4cf7910a7684e5eae36f0b6198cef5a975097cdcd72a25e0f2685a8ead0059")
  message(FATAL_ERROR "Review MiniMax DFlash adapter against changed graph")
endif()
file(READ "${mm27_dflash_original}" content)
set(anchor "    cur = build_lora_mm(output, cur, output_s);\n\n    // DFlash2 feeds these logits")
set(adapter [=[    cur = build_lora_mm(output, cur, output_s);

    // Reviewed MiniMax export borrows a 200064-row head but exposes 200055
    // tokens. A view alone would still have the target row stride: materialize
    // the prefix before llama_context copies n_outputs * draft_vocab floats.
    if (model.output == nullptr && cparams.ctx_other != nullptr &&
            llama_get_model(cparams.ctx_other)->arch == LLM_ARCH_MINIMAX_M2) {
        GGML_ASSERT(model.vocab.n_tokens() == 200055 && cur->ne[0] == 200064);
        GGML_ASSERT(model.d2t == nullptr && model.dflash_selector_hidden == nullptr);
        cb(cur, "mm27_dflash_full_logits", -1);
        cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, 200055, cur->ne[1], cur->nb[1], 0));
    }

    // DFlash2 feeds these logits]=])
mm27_replace_once(content "${anchor}" "${adapter}")
set(generated "${CMAKE_BINARY_DIR}/strata-minimax-m2-dflash.cpp")
set(previous "")
if(EXISTS "${generated}")
  file(READ "${generated}" previous)
endif()
if(NOT previous STREQUAL content)
  file(WRITE "${generated}" "${content}")
endif()
file(SHA256 "${generated}" mm27_dflash_generated_sha256)
get_target_property(sources llama SOURCES)
set(matches "${sources}")
list(FILTER matches INCLUDE REGEX "(^|/)dflash[.]cpp$")
list(LENGTH matches count)
if(NOT count EQUAL 1)
  message(FATAL_ERROR "Expected one DFlash source in llama")
endif()
list(REMOVE_ITEM sources ${matches})
set_property(TARGET llama PROPERTY SOURCES "${sources};${generated}")
set_source_files_properties("${generated}" TARGET_DIRECTORY llama PROPERTIES
  INCLUDE_DIRECTORIES "${mm27_source}/src/models")
string(APPEND mm27_patch_set ",mm27-dflash-borrowed-head-prefix")
