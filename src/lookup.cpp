#include <cmath>
#include <memory>
#include <vector>

#include "trigdx/lookup.hpp"

template <size_t NR_SAMPLES> struct LookupBackend<NR_SAMPLES>::Impl {
  std::vector<float> lookup;
  static constexpr size_t MASK = NR_SAMPLES - 1;
  static constexpr float SCALE = NR_SAMPLES / (2.0f * float(M_PI));

  void init() {
    lookup.resize(NR_SAMPLES);
    for (size_t i = 0; i < NR_SAMPLES; ++i)
      lookup[i] = sinf(i * (2.0f * float(M_PI) / NR_SAMPLES));
  }

  void compute_sinf(size_t n, const float *x, float *s) const {
    for (size_t i = 0; i < n; ++i) {
      size_t idx = static_cast<size_t>(x[i] * SCALE) & MASK;
      s[i] = lookup[idx];
    }
  }

  void compute_cosf(size_t n, const float *x, float *c) const {
    for (size_t i = 0; i < n; ++i) {
      size_t idx = static_cast<size_t>(x[i] * SCALE) & MASK;
      size_t idx_cos = (idx + NR_SAMPLES / 4) & MASK;
      c[i] = lookup[idx_cos];
    }
  }

  void compute_sincosf(size_t n, const float *x, float *s, float *c) const {
    for (size_t i = 0; i < n; ++i) {
      size_t idx = static_cast<size_t>(x[i] * SCALE) & MASK;
      size_t idx_cos = (idx + NR_SAMPLES / 4) & MASK;
      s[i] = lookup[idx];
      c[i] = lookup[idx_cos];
    }
  }
};

template <size_t NR_SAMPLES>
LookupBackend<NR_SAMPLES>::LookupBackend() : impl(std::make_unique<Impl>()) {}

template <size_t NR_SAMPLES>
LookupBackend<NR_SAMPLES>::~LookupBackend() = default;

template <size_t NR_SAMPLES> void LookupBackend<NR_SAMPLES>::init(size_t) {
  impl->init();
}

template <size_t NR_SAMPLES>
void LookupBackend<NR_SAMPLES>::compute_sinf(size_t n, const float *x,
                                             float *s) const {
  impl->compute_sinf(n, x, s);
}

template <size_t NR_SAMPLES>
void LookupBackend<NR_SAMPLES>::compute_cosf(size_t n, const float *x,
                                             float *c) const {
  impl->compute_cosf(n, x, c);
}

template <size_t NR_SAMPLES>
void LookupBackend<NR_SAMPLES>::compute_sincosf(size_t n, const float *x,
                                                float *s, float *c) const {
  impl->compute_sincosf(n, x, s, c);
}

template class LookupBackend<16384>;
template class LookupBackend<32768>;
