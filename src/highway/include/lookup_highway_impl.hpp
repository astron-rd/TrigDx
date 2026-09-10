#pragma once

// Platform-specific definitions used for declaring an interface, independent of
// the SIMD instruction set.
#include "hwy/base.h" // HWY_RESTRICT

namespace highway_impl {

// Computes base-2 logarithm by converting to float. Supports dynamic dispatch.
void compute_sinf(size_t n, const float *HWY_RESTRICT x,
                                const float *HWY_RESTRICT lookup,
                                const size_t mask, const float scale,
                                float *HWY_RESTRICT s);

} // namespace highway_impl
