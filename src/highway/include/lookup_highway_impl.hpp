#pragma once

// Platform-specific definitions used for declaring an interface, independent of
// the SIMD instruction set.
#include "hwy/base.h" // HWY_RESTRICT

namespace highway_impl {

void compute_sinf(size_t n, const float *HWY_RESTRICT x,
                  const float *HWY_RESTRICT lookup, const size_t mask,
                  const float scale, float *HWY_RESTRICT s);

void compute_cosf(size_t n, const float *HWY_RESTRICT x,
                  const float *HWY_RESTRICT lookup, const size_t mask,
                  const float scale, const size_t sample_offset,
                  float *HWY_RESTRICT c);

void compute_sincosf(size_t n, const float *HWY_RESTRICT x,
                     const float *HWY_RESTRICT lookup, const size_t mask,
                     const float scale, const size_t sample_offset,
                     float *HWY_RESTRICT s, float *HWY_RESTRICT c);

} // namespace highway_impl
