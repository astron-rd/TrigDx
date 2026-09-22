#include "include/lookup_highway_impl.hpp"

#include <cstdint>
#include <stdio.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "highway/lookup_highway_impl.cpp"
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

static constexpr float TERM1 = 1.0f;         // 1
static constexpr float TERM2 = 0.5f;         // 1/2!
static constexpr float TERM3 = 1.0f / 6.0f;  // 1/3!
static constexpr float TERM4 = 1.0f / 24.0f; // 1/4!

HWY_ATTR void compute_sinf(size_t n, const T *HWY_RESTRICT x,
                           const T *HWY_RESTRICT lookup, const size_t mask,
                           const float scale, const float pi_frac,
                           const size_t sample_offset, T *HWY_RESTRICT s) {
  const hn::ScalableTag<T> dfloat;
  const hn::ScalableTag<int32_t> dint;

  const auto vscale = hn::Set(dfloat, scale);
  const auto vmask = hn::Set(dint, mask);
  const auto vsample_offset = hn::Set(dint, sample_offset);
  const auto vpi_frac = hn::Set(dfloat, pi_frac);

  const auto term1 = hn::Set(dfloat, TERM1);
  const auto term2 = hn::Set(dfloat, TERM2);

  size_t i = 0;
  for (; i + hn::Lanes(dfloat) <= n; i += hn::Lanes(dfloat)) {
    const auto vx = hn::LoadU(dfloat, &x[i]);
    const auto scaled = hn::Mul(vx, vscale);
    const auto idx = hn::FloorInt(scaled);
    const auto idx_float = hn::ConvertTo(dfloat, idx);
    const auto idx_cos = hn::Add(idx, vsample_offset);
    const auto idx_masked = hn::And(idx, vmask);
    const auto idx_cos_masked = hn::And(idx_cos, vmask);

    const auto dx = hn::Sub(vx, hn::Mul(idx_float, vpi_frac));
    const auto dx2 = hn::Mul(dx, dx);

    const auto t2 = hn::Mul(dx2, term2);

    const auto cosdx = hn::Sub(term1, t2);
    const auto sindx = dx;

    const auto sinv = hn::GatherIndex(dfloat, lookup, idx_masked);
    const auto cosv = hn::GatherIndex(dfloat, lookup, idx_cos_masked);

    const auto sinv_accurate =
        hn::Add(hn::Mul(cosv, sindx), hn::Mul(sinv, cosdx));

    hn::StoreU(sinv_accurate, dfloat, &s[i]);
  }

  for (; i < n; i += 1) {
    const auto scaled = x[i] * scale;
    const auto idx = static_cast<std::size_t>(scaled);
    const auto idx_float = static_cast<float>(idx);
    const auto idx_cos = idx + sample_offset;
    const auto idx_masked = idx & mask;
    const auto idx_cos_masked = idx_cos & mask;

    const auto dx = x[i] - (idx_float * pi_frac);
    const auto dx2 = dx * dx;

    const auto t2 = dx2 * TERM2;

    const auto cosdx = TERM1 - t2;
    const auto sindx = dx;

    const auto sinv = lookup[idx_masked];
    const auto cosv = lookup[idx_cos_masked];

    const auto sinv_accurate = (cosv * sindx) + (sinv * cosdx);

    s[i] = sinv_accurate;
  }
}

HWY_ATTR void compute_cosf(size_t n, const T *HWY_RESTRICT x,
                           const T *HWY_RESTRICT lookup, const size_t mask,
                           const float scale, const float pi_frac,
                           const size_t sample_offset, T *HWY_RESTRICT c) {
  const hn::ScalableTag<T> dfloat;
  const hn::ScalableTag<int32_t> dint;

  const auto vscale = hn::Set(dfloat, scale);
  const auto vmask = hn::Set(dint, mask);
  const auto vsample_offset = hn::Set(dint, sample_offset);
  const auto vpi_frac = hn::Set(dfloat, pi_frac);

  const auto term1 = hn::Set(dfloat, TERM1);
  const auto term2 = hn::Set(dfloat, TERM2);

  size_t i = 0;
  for (; i + hn::Lanes(dfloat) <= n; i += hn::Lanes(dfloat)) {
    const auto vx = hn::LoadU(dfloat, &x[i]);
    const auto scaled = hn::Mul(vx, vscale);
    const auto idx = hn::FloorInt(scaled);
    const auto idx_float = hn::ConvertTo(dfloat, idx);
    const auto idx_cos = hn::Add(idx, vsample_offset);
    const auto idx_masked = hn::And(idx, vmask);
    const auto idx_cos_masked = hn::And(idx_cos, vmask);

    const auto dx = hn::Sub(vx, hn::Mul(idx_float, vpi_frac));
    const auto dx2 = hn::Mul(dx, dx);

    const auto t2 = hn::Mul(dx2, term2);

    const auto cosdx = hn::Sub(term1, t2);
    const auto sindx = dx;

    const auto sinv = hn::GatherIndex(dfloat, lookup, idx_masked);
    const auto cosv = hn::GatherIndex(dfloat, lookup, idx_cos_masked);

    const auto cosv_accurate =
        hn::Sub(hn::Mul(cosv, cosdx), hn::Mul(sinv, sindx));

    hn::StoreU(cosv_accurate, dfloat, &c[i]);
  }

  for (; i < n; i += 1) {
    const auto scaled = x[i] * scale;
    const auto idx = static_cast<std::size_t>(scaled);
    const auto idx_float = static_cast<float>(idx);
    const auto idx_cos = idx + sample_offset;
    const auto idx_masked = idx & mask;
    const auto idx_cos_masked = idx_cos & mask;

    const auto dx = x[i] - (idx_float * pi_frac);
    const auto dx2 = dx * dx;

    const auto t2 = dx2 * TERM2;

    const auto cosdx = TERM1 - t2;
    const auto sindx = dx;

    const auto sinv = lookup[idx_masked];
    const auto cosv = lookup[idx_cos_masked];

    const auto cosv_accurate = (cosv * cosdx) - (sinv * sindx);

    c[i] = cosv_accurate;
  }
}

HWY_ATTR void compute_sincosf(size_t n, const T *HWY_RESTRICT x,
                              const T *HWY_RESTRICT lookup, const size_t mask,
                              const float scale, const float pi_frac,
                              const size_t sample_offset, T *HWY_RESTRICT s,
                              T *HWY_RESTRICT c) {
  const hn::ScalableTag<T> dfloat;
  const hn::ScalableTag<int32_t> dint;

  const auto vscale = hn::Set(dfloat, scale);
  const auto vmask = hn::Set(dint, mask);
  const auto vsample_offset = hn::Set(dint, sample_offset);
  const auto vpi_frac = hn::Set(dfloat, pi_frac);

  const auto term1 = hn::Set(dfloat, TERM1);
  const auto term2 = hn::Set(dfloat, TERM2);

  size_t i = 0;
  for (; i + hn::Lanes(dfloat) <= n; i += hn::Lanes(dfloat)) {
    const auto vx = hn::LoadU(dfloat, &x[i]);
    const auto scaled = hn::Mul(vx, vscale);
    const auto idx = hn::FloorInt(scaled);
    const auto idx_float = hn::ConvertTo(dfloat, idx);
    const auto idx_cos = hn::Add(idx, vsample_offset);
    const auto idx_masked = hn::And(idx, vmask);
    const auto idx_cos_masked = hn::And(idx_cos, vmask);

    const auto dx = hn::Sub(vx, hn::Mul(idx_float, vpi_frac));
    const auto dx2 = hn::Mul(dx, dx);

    const auto t2 = hn::Mul(dx2, term2);

    const auto cosdx = hn::Sub(term1, t2);
    const auto sindx = dx;

    const auto sinv = hn::GatherIndex(dfloat, lookup, idx_masked);
    const auto cosv = hn::GatherIndex(dfloat, lookup, idx_cos_masked);

    const auto sinv_accurate =
        hn::Add(hn::Mul(cosv, sindx), hn::Mul(sinv, cosdx));
    const auto cosv_accurate =
        hn::Sub(hn::Mul(cosv, cosdx), hn::Mul(sinv, sindx));

    hn::StoreU(sinv_accurate, dfloat, &s[i]);
    hn::StoreU(cosv_accurate, dfloat, &c[i]);
  }

  for (; i < n; i += 1) {
    const auto scaled = x[i] * scale;
    const auto idx = static_cast<std::size_t>(scaled);
    const auto idx_float = static_cast<float>(idx);
    const auto idx_cos = idx + sample_offset;
    const auto idx_masked = idx & mask;
    const auto idx_cos_masked = idx_cos & mask;

    const auto dx = x[i] - (idx_float * pi_frac);
    const auto dx2 = dx * dx;

    const auto t2 = dx2 * TERM2;

    const auto cosdx = TERM1 - t2;
    const auto sindx = dx;

    const auto sinv = lookup[idx_masked];
    const auto cosv = lookup[idx_cos_masked];

    const auto sinv_accurate = (cosv * sindx) + (sinv * cosdx);
    const auto cosv_accurate = (cosv * cosdx) - (sinv * sindx);

    s[i] = sinv_accurate;
    c[i] = cosv_accurate;
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
                  const float scale, const float pi_frac,
                  const size_t sample_offset, float *HWY_RESTRICT s) {
  return HWY_DYNAMIC_DISPATCH(compute_sinf)(n, x, lookup, mask, scale, pi_frac,
                                            sample_offset, s);
}

HWY_EXPORT(compute_cosf);

void compute_cosf(size_t n, const float *HWY_RESTRICT x,
                  const float *HWY_RESTRICT lookup, const size_t mask,
                  const float scale, const float pi_frac,
                  const size_t sample_offset, float *HWY_RESTRICT c) {
  return HWY_DYNAMIC_DISPATCH(compute_cosf)(n, x, lookup, mask, scale, pi_frac,
                                            sample_offset, c);
}

HWY_EXPORT(compute_sincosf);

void compute_sincosf(size_t n, const float *HWY_RESTRICT x,
                     const float *HWY_RESTRICT lookup, const size_t mask,
                     const float scale, const float pi_frac,
                     const size_t sample_offset, float *HWY_RESTRICT s,
                     float *HWY_RESTRICT c) {
  return HWY_DYNAMIC_DISPATCH(compute_sincosf)(n, x, lookup, mask, scale,
                                               pi_frac, sample_offset, s, c);
}

} // namespace highway_impl
#endif // HWY_ONCE
