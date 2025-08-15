#include <algorithm>
#include <cmath>
#include <vector>
#include <xsimd/xsimd.hpp>

#include "trigdx/lookup_xsimd.hpp"

template <std::size_t NR_SAMPLES> struct lookup_table {
  static constexpr std::size_t MASK = NR_SAMPLES - 1;
  static constexpr float SCALE = NR_SAMPLES / (2.0f * float(M_PI));
  static constexpr float PI_FRAC = 2.0f * M_PIf32 / NR_SAMPLES;
  static constexpr float TERM1 = 1.0f;         // 1
  static constexpr float TERM2 = 0.5f;         // 1/2!
  static constexpr float TERM3 = 1.0f / 6.0f;  // 1/3!
  static constexpr float TERM4 = 1.0f / 24.0f; // 1/4!

  constexpr lookup_table() : sin_values{}, cos_values{} {
    for (uint_fast32_t i = 0; i < NR_SAMPLES; i++) {
      sin_values[i] = sinf(i * PI_FRAC);
      cos_values[i] = cosf(i * PI_FRAC);
    }
  }
  std::array<float, NR_SAMPLES> cos_values;
  std::array<float, NR_SAMPLES> sin_values;
};

template <std::size_t NR_SAMPLES> struct cosf_dispatcher {
  constexpr cosf_dispatcher() : lookup_table_(){};
  template <class Tag, class Arch>
  void operator()(Arch, const size_t n, const float *a, float *c, Tag) const {
    using b_type = xsimd::batch<float, Arch>;
    using m_type = xsimd::batch<int32_t, Arch>;

    constexpr uint_fast32_t VL = b_type::size;
    const uint_fast32_t VS = n - n % VL;
    const uint_fast32_t Q_PI = NR_SAMPLES / 4U;
    const b_type scale = b_type::broadcast(lookup_table_.SCALE);
    const b_type pi_frac = b_type::broadcast(lookup_table_.PI_FRAC);
    const m_type mask = m_type::broadcast(lookup_table_.MASK);

    const b_type term1 = b_type::broadcast(lookup_table_.TERM1); // 1
    const b_type term2 = b_type::broadcast(lookup_table_.TERM2); // 1/2!
    const b_type term3 = b_type::broadcast(lookup_table_.TERM3); // 1/3!
    const b_type term4 = b_type::broadcast(lookup_table_.TERM4); // 1/4!
    const m_type quarter_pi = m_type::broadcast(Q_PI);
    uint_fast32_t i;
    for (i = 0; i < VS; i += VL) {
      const b_type vx = b_type::load(a + i, Tag());
      const b_type scaled = xsimd::mul(vx, scale);
      m_type idx = xsimd::to_int(scaled);
      const b_type f_idx = xsimd::to_float(idx);
      idx = xsimd::bitwise_and(idx, mask);

      b_type cosv = b_type::gather(lookup_table_.cos_values.data(), idx);
      b_type sinv = b_type::gather(lookup_table_.sin_values.data(), idx);

      const b_type dx = xsimd::sub(vx, xsimd::mul(f_idx, pi_frac));
      const b_type dx2 = xsimd::mul(dx, dx);
      const b_type dx3 = xsimd::mul(dx2, dx);
      const b_type dx4 = xsimd::mul(dx2, dx);
      const b_type t2 = xsimd::mul(dx2, term2);
      const b_type t3 = xsimd::mul(dx3, term3);
      const b_type t4 = xsimd::mul(dx4, term3);

      const b_type cosdx = xsimd::add(xsimd::sub(term1, t2), t4);

      const b_type sindx = xsimd::sub(dx, t3);

      cosv = xsimd::sub(xsimd::mul(cosv, cosdx), xsimd::mul(sinv, sindx));

      cosv.store(c + i, Tag());
    }
    for (; i < n; i++) {
      std::size_t idx = static_cast<std::size_t>(a[i] * lookup_table_.SCALE);

      std::size_t masked = idx & lookup_table_.MASK;
      const float cosv = lookup_table_.cos_values[masked];
      const float sinv = lookup_table_.sin_values[masked];
      const float dx = a[i] - idx * lookup_table_.PI_FRAC;
      const float dx2 = dx * dx;
      const float dx3 = dx2 * dx;
      const float dx4 = dx3 * dx;
      const float cosdx =
          1.0f - lookup_table_.TERM2 * dx2 + lookup_table_.TERM4 * dx4;
      const float sindx = dx - lookup_table_.TERM3 * dx3;
      c[i] = cosv * cosdx - sinv * sindx;
    }
  }
  lookup_table<NR_SAMPLES> lookup_table_;
};

template <std::size_t NR_SAMPLES> struct sinf_dispatcher {
  constexpr sinf_dispatcher() : lookup_table_(){};
  template <class Tag, class Arch>
  void operator()(Arch, const size_t n, const float *a, float *s, Tag) const {
    using b_type = xsimd::batch<float, Arch>;
    using m_type = xsimd::batch<int32_t, Arch>;

    constexpr uint_fast32_t VL = b_type::size;
    const uint_fast32_t VS = n - n % VL;
    const uint_fast32_t Q_PI = NR_SAMPLES / 4U;
    const b_type scale = b_type::broadcast(lookup_table_.SCALE);
    const b_type pi_frac = b_type::broadcast(lookup_table_.PI_FRAC);
    const m_type mask = m_type::broadcast(lookup_table_.MASK);

    const b_type term1 = b_type::broadcast(lookup_table_.TERM1); // 1
    const b_type term2 = b_type::broadcast(lookup_table_.TERM2); // 1/2!
    const b_type term3 = b_type::broadcast(lookup_table_.TERM3); // 1/3!
    const b_type term4 = b_type::broadcast(lookup_table_.TERM4); // 1/4!
    const m_type quarter_pi = m_type::broadcast(Q_PI);
    uint_fast32_t i;
    for (i = 0; i < VS; i += VL) {
      const b_type vx = b_type::load(a + i, Tag());
      const b_type scaled = xsimd::mul(vx, scale);
      m_type idx = xsimd::to_int(scaled);
      b_type f_idx = xsimd::to_float(idx);
      const b_type dx = xsimd::sub(vx, xsimd::mul(f_idx, pi_frac));
      const b_type dx2 = xsimd::mul(dx, dx);
      const b_type dx3 = xsimd::mul(dx2, dx);
      const b_type dx4 = xsimd::mul(dx2, dx);
      const b_type t2 = xsimd::mul(dx2, term2);
      const b_type t3 = xsimd::mul(dx3, term3);
      const b_type t4 = xsimd::mul(dx4, term3);

      const b_type cosdx = xsimd::add(xsimd::sub(term1, t2), t4);
      const b_type sindx = xsimd::sub(dx, t3);

      idx = xsimd::bitwise_and(idx, mask);
      b_type sinv = b_type::gather(lookup_table_.sin_values.data(), idx);
      const b_type cosv = b_type::gather(lookup_table_.cos_values.data(), idx);

      sinv = xsimd::add(xsimd::mul(cosv, sindx), xsimd::mul(sinv, cosdx));
      sinv.store(s + i, Tag());
    }
    for (; i < n; i++) {
      std::size_t idx = static_cast<std::size_t>(a[i] * lookup_table_.SCALE);
      std::size_t masked = idx & lookup_table_.MASK;
      const float cosv = lookup_table_.cos_values[masked];
      const float sinv = lookup_table_.sin_values[masked];
      const float dx = a[i] - idx * lookup_table_.PI_FRAC;
      const float dx2 = dx * dx;
      const float dx3 = dx2 * dx;
      const float dx4 = dx3 * dx;
      const float cosdx =
          1.0f - lookup_table_.TERM2 * dx2 + lookup_table_.TERM4 * dx4;
      const float sindx = dx - lookup_table_.TERM3 * dx3;

      s[i] = sinv * cosdx + cosv * sindx;
    }
  }
  lookup_table<NR_SAMPLES> lookup_table_;
};

template <std::size_t NR_SAMPLES> struct sin_cosf_dispatcher {
  template <class Tag, class Arch>
  void operator()(Arch, const size_t n, const float *a, float *s, float *c,
                  Tag) const {
    using b_type = xsimd::batch<float, Arch>;
    using m_type = xsimd::batch<int32_t, Arch>;

    constexpr uint_fast32_t VL = b_type::size;
    const uint_fast32_t VS = n - n % VL;
    const uint_fast32_t Q_PI = NR_SAMPLES / 4U;
    const b_type scale = b_type::broadcast(lookup_table_.SCALE);
    const m_type mask = m_type::broadcast(lookup_table_.MASK);
    const b_type pi_frac = b_type::broadcast(lookup_table_.PI_FRAC);

    const b_type term1 = b_type::broadcast(lookup_table_.TERM1); // 1
    const b_type term2 = b_type::broadcast(lookup_table_.TERM2); // 1/2!
    const b_type term3 = b_type::broadcast(lookup_table_.TERM3); // 1/3!
    const b_type term4 = b_type::broadcast(lookup_table_.TERM4); // 1/4!

    const m_type quarter_pi = m_type::broadcast(Q_PI);
    uint_fast32_t i;
    for (i = 0; i < VS; i += VL) {
      const b_type vx = b_type::load(a + i, Tag());
      const b_type scaled = xsimd::mul(vx, scale);
      m_type idx = xsimd::to_int(scaled);
      b_type f_idx = xsimd::to_float(idx);
      const b_type dx = xsimd::sub(vx, xsimd::mul(f_idx, pi_frac));
      const b_type dx2 = xsimd::mul(dx, dx);
      const b_type dx3 = xsimd::mul(dx2, dx);
      const b_type dx4 = xsimd::mul(dx2, dx);
      const b_type t2 = xsimd::mul(dx2, term2);
      const b_type t3 = xsimd::mul(dx3, term3);
      const b_type t4 = xsimd::mul(dx4, term3);

      idx = xsimd::bitwise_and(idx, mask);
      b_type sinv = b_type::gather(lookup_table_.sin_values.data(), idx);
      b_type cosv = b_type::gather(lookup_table_.cos_values.data(), idx);

      const b_type cosdx = xsimd::add(xsimd::sub(term1, t2), t4);
      const b_type sindx = xsimd::sub(dx, t3);

      sinv = xsimd::add(xsimd::mul(cosv, sindx), xsimd::mul(sinv, cosdx));
      cosv = xsimd::sub(xsimd::mul(cosv, cosdx), xsimd::mul(sinv, sindx));

      sinv.store(s + i, Tag());
      cosv.store(c + i, Tag());
    }
    for (; i < n; i++) {
      std::size_t idx = static_cast<std::size_t>(a[i] * lookup_table_.SCALE);
      std::size_t masked = idx & lookup_table_.MASK;
      const float cosv = lookup_table_.cos_values[masked];
      const float sinv = lookup_table_.sin_values[masked];
      const float dx = a[i] - idx * lookup_table_.PI_FRAC;
      const float dx2 = dx * dx;
      const float dx3 = dx2 * dx;
      const float dx4 = dx3 * dx;
      const float cosdx =
          1.0f - lookup_table_.TERM2 * dx2 + lookup_table_.TERM4 * dx4;
      const float sindx = dx - lookup_table_.TERM3 * dx3;
      s[i] = sinv * cosdx + cosv * sindx;
      c[i] = cosv * cosdx - sinv * sindx;
    }
  }
  lookup_table<NR_SAMPLES> lookup_table_;
};

template <std::size_t NR_SAMPLES> struct LookupXSIMDBackend<NR_SAMPLES>::Impl {
  cosf_dispatcher<NR_SAMPLES> cosf_dispatcher_;
  sinf_dispatcher<NR_SAMPLES> sinf_dispatcher_;
  sin_cosf_dispatcher<NR_SAMPLES> sin_cosf_dispatcher_;

  void init() {}

  void compute_sincosf(std::size_t n, const float *x, float *s,
                       float *c) const {
    xsimd::dispatch(sin_cosf_dispatcher_)(n, x, s, c, xsimd::unaligned_mode());
  }

  void compute_sinf(std::size_t n, const float *x, float *s) const {
    xsimd::dispatch(sinf_dispatcher_)(n, x, s, xsimd::unaligned_mode());
  }

  void compute_cosf(std::size_t n, const float *x, float *c) const {
    xsimd::dispatch(cosf_dispatcher_)(n, x, c, xsimd::unaligned_mode());
  }
};

template <std::size_t NR_SAMPLES>
LookupXSIMDBackend<NR_SAMPLES>::LookupXSIMDBackend()
    : impl(std::make_unique<Impl>()) {}

template <std::size_t NR_SAMPLES>
LookupXSIMDBackend<NR_SAMPLES>::~LookupXSIMDBackend() = default;

template <std::size_t NR_SAMPLES>
void LookupXSIMDBackend<NR_SAMPLES>::init(size_t) {
  impl->init();
}

template <std::size_t NR_SAMPLES>
void LookupXSIMDBackend<NR_SAMPLES>::compute_sinf(const std::size_t n,
                                                  const float *x,
                                                  float *s) const {
  impl->compute_sinf(n, x, s);
}

template <std::size_t NR_SAMPLES>
void LookupXSIMDBackend<NR_SAMPLES>::compute_cosf(const std::size_t n,
                                                  const float *x,
                                                  float *c) const {
  impl->compute_cosf(n, x, c);
}

template <std::size_t NR_SAMPLES>
void LookupXSIMDBackend<NR_SAMPLES>::compute_sincosf(const std::size_t n,
                                                     const float *x, float *s,
                                                     float *c) const {
  impl->compute_sincosf(n, x, s, c);
}

// Explicit instantiations
template class LookupXSIMDBackend<16384>;
template class LookupXSIMDBackend<32768>;
