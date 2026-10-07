#pragma once
#include "ggml.h"
#include <cstdlib>
#include <cstring>

inline bool mimo2_target_head_columns_enabled() {
    const char * value = std::getenv("STRATA_MIMO_TARGET_HEAD_COLUMNS");
    return value && std::strcmp(value, "1") == 0;
}

// Preserve the ordinary one-column GPU projection. The caller retains the
// model's weight/scaling/LoRA path; only the activation views are separated.
template<class Project>
ggml_tensor * mimo2_project_head_columns(ggml_context * ctx, ggml_tensor * input, Project project) {
    GGML_ASSERT(input->type == GGML_TYPE_F32 && input->ne[1] >= 1 && input->ne[1] <= 8);
    GGML_ASSERT(input->ne[2] == 1 && input->ne[3] == 1);
    ggml_tensor * output = nullptr;
    for (int64_t i = 0; i < input->ne[1]; ++i) {
        auto * column = ggml_view_2d(ctx, input, input->ne[0], 1, input->nb[1], i * input->nb[1]);
        auto * projected = project(column);
        output = output ? ggml_concat(ctx, output, projected, 1) : projected;
    }
    return output;
}
