#pragma once
#include <cstdlib>
#include <cstring>

// Startup setting in production. Tests may switch it only with CUDA graphs
// disabled. 0: upstream, 1: routed matrices, 2: routed and dense matrices.
static inline int strata_ds4_mmvq_mode() {
    const char *v=std::getenv("STRATA_DS4_MMVQ_TOKEN_BATCH");
    return v && std::strcmp(v,"1")==0?1:v && std::strcmp(v,"2")==0?2:0;
}
