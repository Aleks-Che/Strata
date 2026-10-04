#pragma once
#include <cstddef>

// Immutable source ranges only. The caller owns mapped memory until finish();
// native jobs separately retain their registered file handles. Destination
// tensors can alias activations and must be written at the normal copy point.
struct StrataExpertSlice { const void *data; size_t bytes; bool cacheable; };
