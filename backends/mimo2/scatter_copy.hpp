#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>

// Device pointers only; destinations must be disjoint from each other and all
// sources. Caller retains buffers until stream completion. Metadata is copied
// into kernel launch parameters; no device allocation or async host staging.
// tile_kib is a diagnostic tuning argument, not a runtime environment setting.
cudaError_t mimo_scatter_copy(void *const *dsts,const void *const *srcs,const size_t *sizes,
    size_t count,cudaStream_t stream,size_t *launches=nullptr,int tile_kib=16);
