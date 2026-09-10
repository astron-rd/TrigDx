#include "/home/viv/Code/TrigDx/src/highway/include/lookup_highway_impl.hpp"

#include <cstdint>
#include <stdio.h>

// >>>> for dynamic dispatch only, skip if you want static dispatch

// First undef to prevent error when re-included.
#undef HWY_TARGET_INCLUDE
// For dynamic dispatch, specify the name of the current file (unfortunately
// __FILE__ is not reliable) so that foreach_target.h can re-include it.
#define HWY_TARGET_INCLUDE                                                     \
  "/home/viv/Code/TrigDx/src/highway/lookup_highway_impl.cpp"
// Generates code for each enabled target by re-including this source file.
#include "hwy/foreach_target.h" // IWYU pragma: keep

// <<<< end of dynamic dispatch

// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/highway.h"

namespace highway_impl {
// This namespace name is unique per target, which allows code for multiple
// targets to co-exist in the same translation unit. Required when using dynamic
// dispatch, otherwise optional.
namespace HWY_NAMESPACE {

// Highway ops reside here; ADL does not find templates nor builtins.
namespace hn = hwy::HWY_NAMESPACE;

using T = float;

HWY_ATTR void compute_sinf(size_t n, const T *HWY_RESTRICT x,
                           const T *HWY_RESTRICT lookup, const size_t mask,
                           const float scale, T *HWY_RESTRICT s) {
  const hn::ScalableTag<T> dfloat;
  const hn::ScalableTag<int32_t> dint;

  const auto vscale = hn::Set(dfloat, scale);
  const auto vmask = hn::Set(dint, mask);

  for (size_t i = 0; i < n; i += hn::Lanes(dfloat)) {
    const auto vx = hn::LoadU(dfloat, &x[i]);
    const auto scaled = hn::Mul(vx, vscale);
    const auto idx = hn::NearestInt(scaled);
    const auto idx_masked = hn::And(idx, vmask);

    const auto sinv = hn::GatherIndex(dfloat, lookup, idx_masked);

    hn::StoreU(sinv, dfloat, &s[i]);
  }
}

} // namespace HWY_NAMESPACE
} // namespace highway_impl

#if HWY_ONCE

namespace highway_impl {

// This macro declares a static array used for dynamic dispatch; it resides in
// the same outer namespace that contains FloorLog2.
HWY_EXPORT(compute_sinf);

void compute_sinf(size_t n, const float *HWY_RESTRICT x,
                                const float *HWY_RESTRICT lookup, const size_t mask,
                                const float scale, float *HWY_RESTRICT s) {
  return HWY_DYNAMIC_DISPATCH(compute_sinf)(n, x, lookup, mask, scale, s);
}

// Optional: anything to compile only once, e.g. non-SIMD implementations of
// public functions provided by this module, can go inside #if HWY_ONCE.

} // namespace highway_impl
#endif // HWY_ONCE
