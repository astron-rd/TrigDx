#include <algorithm>
#include <cmath>
#include <vector>
#include <xsimd/xsimd.hpp>

#include "trigdx/lookup_xsimd.hpp"

template <std::size_t NR_SAMPLES> struct lookup_table {
  static constexpr std::size_t MASK = NR_SAMPLES - 1;
  static constexpr float SCALE = NR_SAMPLES / (2.0f * float(M_PI));
<<<<<<< HEAD
  lookup_table() : values{} {
    for (uint_fast32_t i = 0; i < NR_SAMPLES; i++) {
      values[i] = sinf(i * (2.0f * float(M_PI) / NR_SAMPLES));
    }
  }
  std::array<float, NR_SAMPLES> values;
=======
  lookup_table() : sin_values{}, cos_values{} {
    constexpr float PI_FRAC = 2.0f * M_PIf32 / NR_SAMPLES;
    for (uint_fast32_t i = 0; i < NR_SAMPLES; i++) {
      sin_values[i] = sinf(i * PI_FRAC);
      cos_values[i] = cosf(i * PI_FRAC);
    }
  }
  std::array<float, NR_SAMPLES> cos_values;
  std::array<float, NR_SAMPLES> sin_values;
>>>>>>> 9772c9c (Add xsimd taylor expansion)
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
    const m_type mask = m_type::broadcast(lookup_table_.MASK);
<<<<<<< HEAD
=======
    const b_type term1 = b_type::broadcast(1.0f);         // 1
    const b_type term2 = b_type::broadcast(0.5f);         // 1/2!
    const b_type term3 = b_type::broadcast(1.0f / 6.0f);  // 1/3!
    const b_type term4 = b_type::broadcast(1.0f / 24.0f); // 1/4!
>>>>>>> 9772c9c (Add xsimd taylor expansion)

    const m_type quarter_pi = m_type::broadcast(Q_PI);
    uint_fast32_t i;
    for (i = 0; i < VS; i += VL) {
      const b_type vx = b_type::load(a + i, Tag());
      const b_type scaled = xsimd::mul(vx, scale);
      m_type idx = xsimd::to_int(scaled);
<<<<<<< HEAD
      m_type idx_cos = xsimd::add(idx, quarter_pi);
      idx_cos = xsimd::bitwise_and(idx_cos, mask);
      const b_type cosv = b_type::gather(lookup_table_.values.data(), idx_cos);
=======
      idx = xsimd::bitwise_and(idx, mask);
      const b_type f_idx = xsimd::to_float(idx);

      b_type cosv = b_type::gather(lookup_table_.cos_values.data(), idx);
      b_type sinv = b_type::gather(lookup_table_.sin_values.data(), idx);

      const b_type dx = xsimd::sub(vx, xsimd::mul(f_idx, scale));
      const b_type cosdx =
          term1 - (term2 * dx * dx) + (term4 * dx * dx * dx * dx);
      const b_type sindx = dx - (term3 * dx * dx * dx);

      cosv = cosv * cosdx - sinv * sindx;
>>>>>>> 9772c9c (Add xsimd taylor expansion)

      cosv.store(c + i, Tag());
    }
    for (; i < n; i++) {
      std::size_t idx = static_cast<std::size_t>(a[i] * lookup_table_.SCALE) &
                        lookup_table_.MASK;
<<<<<<< HEAD
      std::size_t idx_cos = (idx + Q_PI) & lookup_table_.MASK;

      c[i] = lookup_table_.values[idx_cos];
=======

      c[i] = lookup_table_.cos_values[idx];
>>>>>>> 9772c9c (Add xsimd taylor expansion)
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
    const m_type mask = m_type::broadcast(lookup_table_.MASK);

    const m_type quarter_pi = m_type::broadcast(Q_PI);
    uint_fast32_t i;
    for (i = 0; i < VS; i += VL) {
      const b_type vx = b_type::load(a + i, Tag());
      const b_type scaled = xsimd::mul(vx, scale);
      m_type idx = xsimd::to_int(scaled);
      idx = xsimd::bitwise_and(idx, mask);
<<<<<<< HEAD
      const b_type sinv = b_type::gather(lookup_table_.values.data(), idx);
=======
      const b_type sinv = b_type::gather(lookup_table_.sin_values.data(), idx);
>>>>>>> 9772c9c (Add xsimd taylor expansion)

      sinv.store(s + i, Tag());
    }
    for (; i < n; i++) {
      std::size_t idx = static_cast<std::size_t>(a[i] * lookup_table_.SCALE) &
                        lookup_table_.MASK;
<<<<<<< HEAD
      s[i] = lookup_table_.values[idx];
=======
      s[i] = lookup_table_.sin_values[idx];
>>>>>>> 9772c9c (Add xsimd taylor expansion)
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

    const m_type quarter_pi = m_type::broadcast(Q_PI);
    uint_fast32_t i;
    for (i = 0; i < VS; i += VL) {
      const b_type vx = b_type::load(a + i, Tag());
      const b_type scaled = xsimd::mul(vx, scale);
      m_type idx = xsimd::to_int(scaled);
      m_type idx_cos = xsimd::add(idx, quarter_pi);
      idx = xsimd::bitwise_and(idx, mask);
<<<<<<< HEAD
      idx_cos = xsimd::bitwise_and(idx_cos, mask);
      const b_type sinv = b_type::gather(lookup_table_.values.data(), idx);
      const b_type cosv = b_type::gather(lookup_table_.values.data(), idx_cos);
=======
      const b_type sinv = b_type::gather(lookup_table_.sin_values.data(), idx);
      const b_type cosv = b_type::gather(lookup_table_.cos_values.data(), idx);
>>>>>>> 9772c9c (Add xsimd taylor expansion)

      sinv.store(s + i, Tag());
      cosv.store(c + i, Tag());
    }
    for (; i < n; i++) {
      std::size_t idx = static_cast<std::size_t>(a[i] * lookup_table_.SCALE) &
                        lookup_table_.MASK;
<<<<<<< HEAD
      std::size_t idx_cos = (idx + Q_PI) & lookup_table_.MASK;
      s[i] = lookup_table_.values[idx];
      c[i] = lookup_table_.values[idx_cos];
=======
      s[i] = lookup_table_.cos_values[idx];
      c[i] = lookup_table_.sin_values[idx];
>>>>>>> 9772c9c (Add xsimd taylor expansion)
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
