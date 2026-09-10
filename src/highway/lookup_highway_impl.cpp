#include "include/lookup_highway_impl.hpp"

#include <cstdint>
#include <stdio.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "../../../src/highway/lookup_highway_impl.cpp"
#include "hwy/foreach_target.h" // IWYU pragma: keep

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

  size_t i = 0;
  for (; i + hn::Lanes(dfloat) <= n; i += hn::Lanes(dfloat)) {
    const auto vx = hn::LoadU(dfloat, &x[i]);
    const auto scaled = hn::Mul(vx, vscale);
    const auto idx = hn::FloorInt(scaled);
    const auto idx_masked = hn::And(idx, vmask);

    const auto sinv = hn::GatherIndex(dfloat, lookup, idx_masked);

    hn::StoreU(sinv, dfloat, &s[i]);
  }

  for (; i < n; i += 1) {
    std::size_t idx = static_cast<std::size_t>(x[i] * scale) & mask;
    s[i] = lookup[idx];
  }
}

HWY_ATTR void compute_cosf(size_t n, const T *HWY_RESTRICT x,
                           const T *HWY_RESTRICT lookup, const size_t mask,
                           const float scale, const size_t sample_offset,
                           T *HWY_RESTRICT c) {
  const hn::ScalableTag<T> dfloat;
  const hn::ScalableTag<int32_t> dint;

  const auto vscale = hn::Set(dfloat, scale);
  const auto vmask = hn::Set(dint, mask);
  const auto vsample_offset = hn::Set(dint, sample_offset);

  size_t i = 0;
  for (; i + hn::Lanes(dfloat) <= n; i += hn::Lanes(dfloat)) {
    const auto vx = hn::LoadU(dfloat, &x[i]);
    const auto scaled = hn::Mul(vx, vscale);
    const auto idx = hn::FloorInt(scaled);
    const auto idx_cos = hn::Add(idx, vsample_offset);
    const auto idx_masked = hn::And(idx_cos, vmask);

    const auto cosv = hn::GatherIndex(dfloat, lookup, idx_masked);

    hn::StoreU(cosv, dfloat, &c[i]);
  }

  for (; i < n; i += 1) {
    std::size_t idx = static_cast<std::size_t>(x[i] * scale);
    std::size_t idx_cos = (idx + sample_offset) & mask;
    c[i] = lookup[idx_cos];
  }
}

HWY_ATTR void compute_sincosf(size_t n, const T *HWY_RESTRICT x,
                              const T *HWY_RESTRICT lookup, const size_t mask,
                              const float scale, const size_t sample_offset,
                              T *HWY_RESTRICT s, T *HWY_RESTRICT c) {
  const hn::ScalableTag<T> dfloat;
  const hn::ScalableTag<int32_t> dint;

  const auto vscale = hn::Set(dfloat, scale);
  const auto vmask = hn::Set(dint, mask);
  const auto vsample_offset = hn::Set(dint, sample_offset);

  size_t i = 0;
  for (; i + hn::Lanes(dfloat) <= n; i += hn::Lanes(dfloat)) {
    const auto vx = hn::LoadU(dfloat, &x[i]);
    const auto scaled = hn::Mul(vx, vscale);
    const auto idx = hn::FloorInt(scaled);
    const auto idx_masked = hn::And(idx, vmask);

    const auto idx_cos = hn::Add(idx, vsample_offset);
    const auto idx_cos_masked = hn::And(idx_cos, vmask);

    const auto sinv = hn::GatherIndex(dfloat, lookup, idx_masked);
    const auto cosv = hn::GatherIndex(dfloat, lookup, idx_cos_masked);

    hn::StoreU(sinv, dfloat, &s[i]);
    hn::StoreU(cosv, dfloat, &c[i]);
  }

  for (; i < n; i += 1) {
    std::size_t idx = static_cast<std::size_t>(x[i] * scale) & mask;
    std::size_t idx_cos = (idx + sample_offset) & mask;
    s[i] = lookup[idx];
    c[i] = lookup[idx_cos];
  }
}

} // namespace HWY_NAMESPACE
} // namespace highway_impl

#if HWY_ONCE

namespace highway_impl {

// This macro declares a static array used for dynamic dispatch
HWY_EXPORT(compute_sinf);

void compute_sinf(size_t n, const float *HWY_RESTRICT x,
                  const float *HWY_RESTRICT lookup, const size_t mask,
                  const float scale, float *HWY_RESTRICT s) {
  return HWY_DYNAMIC_DISPATCH(compute_sinf)(n, x, lookup, mask, scale, s);
}

HWY_EXPORT(compute_cosf);

void compute_cosf(size_t n, const float *HWY_RESTRICT x,
                  const float *HWY_RESTRICT lookup, const size_t mask,
                  const float scale, const size_t sample_offset,
                  float *HWY_RESTRICT c) {
  return HWY_DYNAMIC_DISPATCH(compute_cosf)(n, x, lookup, mask, scale,
                                            sample_offset, c);
}

HWY_EXPORT(compute_sincosf);

void compute_sincosf(size_t n, const float *HWY_RESTRICT x,
                     const float *HWY_RESTRICT lookup, const size_t mask,
                     const float scale, const size_t sample_offset,
                     float *HWY_RESTRICT s, float *HWY_RESTRICT c) {
  return HWY_DYNAMIC_DISPATCH(compute_sincosf)(n, x, lookup, mask, scale,
                                               sample_offset, s, c);
}

} // namespace highway_impl
#endif // HWY_ONCE
