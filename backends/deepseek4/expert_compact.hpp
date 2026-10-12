#pragma once
#include "ggml.h"
#include <cstdint>

// Private scheduler copies only. Logical expert IDs/shape stay unchanged;
// MMVQ's existing pointer table maps them to densely packed physical slots.
// No other operation is permitted to read this undersized allocation.
inline int strata_compact_slots(const ggml_tensor *t) {
    return t && t->op==GGML_OP_NONE && !t->view_src && t->op_params[0]==0x53444350 &&
        t->op_params[1]>0 && t->op_params[1]<t->ne[2] ? t->op_params[1]:0;
}
inline size_t strata_compact_bytes(const ggml_tensor *t) {
    const int slots=strata_compact_slots(t);
    return slots?size_t(slots)*(t->nb[2]+512):0;
}
