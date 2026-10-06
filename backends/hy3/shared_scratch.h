#pragma once
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif
// Private Hy3 APIs. Share only freshly reserved CUDA0 compute buffers, before
// allocating an executable graph. Callers must serialize both contexts and
// refresh every graph input after switching contexts. KV/state stay separate.
// Returns physical bytes released, or zero if the layout cannot be shared.
size_t strata_hy3_gallocr_share_scratch(ggml_gallocr_t a, ggml_gallocr_t b);
size_t strata_hy3_sched_share_scratch(ggml_backend_sched_t a, ggml_backend_sched_t b);
bool strata_hy3_gallocr_scratch_is_shared(ggml_gallocr_t a, ggml_gallocr_t b);
bool strata_hy3_sched_scratch_is_shared(ggml_backend_sched_t a, ggml_backend_sched_t b);
#ifdef __cplusplus
}
#endif
