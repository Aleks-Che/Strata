# Experimental comparison only; private generated sources, original pin untouched.
# MTP-only sidecars load head0; the two later offset heads are intentionally skipped.
function(mimo_spec_source name hash)
  set(original "${mimo_source}/src/models/${name}.cpp")
  file(SHA256 "${original}" actual)
  if(NOT actual STREQUAL hash)
    message(FATAL_ERROR "Review speculative patch for ${name}")
  endif()
  file(READ "${original}" content)
  if(name STREQUAL "mimo2")
    string(PREPEND content "#include <algorithm>\n#include \"target_head.hpp\"\n")
    mimo_replace_once(content "    int mtp_flags         = trunk_only ? TENSOR_NOT_REQUIRED : 0;"
      "    int mtp_flags         = trunk_only ? TENSOR_NOT_REQUIRED : 0;\n    const bool mtp_only = hparams.n_layer_nextn > 0 && ml.get_weight(\"blk.0.attn_norm.weight\") == nullptr;")
    mimo_replace_once(content "const int  flags    = is_nextn ? mtp_flags : 0;"
      "const int  flags    = is_nextn ? (mtp_flags | (mtp_only && i > n_layer ? TENSOR_SKIP : 0)) : (mtp_only ? TENSOR_NOT_REQUIRED : 0);")
    mimo_replace_once(content "    const bool crop_last_layer = inp_out_ids && (!emit_h_nextn || cparams.embeddings_nextn_masked);"
      "    const bool features = std::any_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(), [](bool b) {return b;});\n    const bool crop_last_layer = !features && inp_out_ids && (!emit_h_nextn || cparams.embeddings_nextn_masked);")
    mimo_replace_once(content "    for (int il = 0; il < n_layer; ++il) {\n        ggml_tensor * inpSA = inpL;"
      "    for (int il = 0; il < n_layer; ++il) {\n        res->t_layer_inp[il] = inpL;\n        ggml_tensor * inpSA = inpL;")
    mimo_replace_once(content "    cur = inpL;\n\n    if (emit_h_nextn) {"
      "    cur = inpL;\n    res->t_layer_inp[n_layer] = inpL;\n    if (features && !emit_h_nextn && inp_out_ids) cur = ggml_get_rows(ctx0, cur, inp_out_ids);\n\n    if (emit_h_nextn) {")
    mimo_replace_once(content "    cur = build_lora_mm(model.output, cur, model.output_s);"
      "    if (mimo2_target_head_columns_enabled() && model.output->type == GGML_TYPE_BF16 &&\n        cur->type == GGML_TYPE_F32 && cur->ne[1] > 1 && cur->ne[1] <= 8 && cur->ne[2] == 1 && cur->ne[3] == 1) {\n        cur = mimo2_project_head_columns(ctx0, cur, [&](ggml_tensor * column) {\n            return build_lora_mm(model.output, column, model.output_s);\n        });\n    } else {\n        cur = build_lora_mm(model.output, cur, model.output_s);\n    }")
  else()
    string(PREPEND content "#include <cstdlib>\n")
    set(load_anchor "void llama_model_dflash::load_arch_tensors(llama_model_loader &) {\n    LLAMA_LOAD_LOCALS;")
    mimo_replace_once(content "${load_anchor}"
      "${load_anchor}\n    const bool shared_target = std::getenv(\"STRATA_MIMO_DFLASH_SHARE_TARGET\") && std::string(std::getenv(\"STRATA_MIMO_DFLASH_SHARE_TARGET\")) == \"1\";\n    const int shared_flags = TENSOR_NOT_REQUIRED | (shared_target ? TENSOR_SKIP : 0);")
    mimo_replace_once(content "{ n_embd, n_vocab }, TENSOR_NOT_REQUIRED);"
      "{ n_embd, n_vocab }, shared_flags);")
    mimo_replace_once(content "{ n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);"
      "{ n_embd, n_vocab_draft }, shared_flags);")
    mimo_replace_once(content "    hparams.f_final_logit_softcapping = 0.0f;"
      "    ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, hparams.f_attn_value_scale, false);\n    hparams.f_final_logit_softcapping = 0.0f;")
    mimo_replace_once(content "        if (attn_dynamic) {\n            cur = build_dflash2_conv"
      "        if (hparams.f_attn_value_scale != 0.0f) {\n            cur = ggml_scale(ctx0, cur, hparams.f_attn_value_scale);\n        }\n        if (attn_dynamic) {\n            cur = build_dflash2_conv")
    mimo_replace_once(content "    cur = build_lora_mm(output, cur, output_s);\n\n    // DFlash2 feeds these logits"
      "    if (cparams.ctx_other && model.tok_embd && model.tok_embd->ne[1] == 1) {\n        // Small blocks: avoid a second full BF16-head-to-F32 conversion pool\n        // in the draft context. Each column uses the ordinary GPU vector path.\n        auto * normalized = cur;\n        ggml_tensor * logits = nullptr;\n        for (int64_t i = 0; i < n_tokens; ++i) {\n            auto * column = ggml_view_2d(ctx0, normalized, n_embd, 1, normalized->nb[1], i * normalized->nb[1]);\n            auto * projected = build_lora_mm(output, column, output_s);\n            logits = logits ? ggml_concat(ctx0, logits, projected, 1) : projected;\n        }\n        cur = logits;\n    } else {\n        cur = build_lora_mm(output, cur, output_s);\n    }\n\n    // DFlash2 feeds these logits")
    set(embedding_anchor "    res->t_inp_tokens = inp->tokens;\n\n    ggml_tensor * inp_tokens = inp->tokens;\n\n")
    mimo_replace_once(content "${embedding_anchor}    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);"
      "${embedding_anchor}    ggml_tensor * inpL;\n    if (tok_embd->ne[1] == 1 && cparams.ctx_other) {\n        // Probe owns a separate learned MASK row; target weights stay immutable.\n        const auto * target = llama_get_model(cparams.ctx_other);\n        auto * embedded = ggml_get_rows(ctx0, target->tok_embd, inp->tokens);\n        inpL = ggml_view_2d(ctx0, embedded, n_embd, 1, embedded->nb[1], 0);\n        if (n_tokens > 1) {\n            auto * masks = ggml_repeat_4d(ctx0, tok_embd, n_embd, n_tokens - 1, 1, 1);\n            inpL = ggml_concat(ctx0, inpL, masks, 1);\n        }\n    } else {\n        inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);\n    }")
  endif()
  set(output "${CMAKE_BINARY_DIR}/strata-mimo2-spec-${name}.cpp")
  set(previous "")
  if(EXISTS "${output}")
    file(READ "${output}" previous)
  endif()
  if(NOT previous STREQUAL content)
    file(WRITE "${output}" "${content}")
  endif()
  file(SHA256 "${output}" generated_hash)
  set(mimo_spec_hashes "${mimo_spec_hashes}${name}:${generated_hash};" PARENT_SCOPE)
  get_target_property(sources llama SOURCES)
  set(matches "${sources}")
  list(FILTER matches INCLUDE REGEX "(^|/)${name}\\.cpp$")
  list(LENGTH matches count)
  if(NOT count EQUAL 1)
    message(FATAL_ERROR "Expected one ${name} source")
  endif()
  list(REMOVE_ITEM sources ${matches})
  set_property(TARGET llama PROPERTY SOURCES "${sources};${output}")
  set_source_files_properties("${output}" TARGET_DIRECTORY llama PROPERTIES
    INCLUDE_DIRECTORIES "${mimo_source}/src/models;${CMAKE_CURRENT_SOURCE_DIR}"
    OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/target_head.hpp")
endfunction()
mimo_spec_source(mimo2 "${STRATA_MIMO_LOADER_SHA256}")
file(SHA256 "${mimo_source}/src/models/dflash.cpp" mimo_dflash_hash)
# The archive hash pins this source in addition to its recorded generated hash.
mimo_spec_source(dflash "${mimo_dflash_hash}")
string(APPEND mimo_patch_set ",mimo-experimental-draft-features-sidecar-dflash-vscale,optional-target-head-columns")
