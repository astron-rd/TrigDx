#include <cmath>
#include <vector>

#include "trigdx/lookup_highway.hpp"
#include "highway/include/lookup_highway_impl.hpp"

#if defined(HAVE_AVX) && !defined(__AVX__)
static_assert(HAVE_AVX == 0, "__AVX__ should be defined when HAVE_AVX is "
                             "defined");
#endif

#if defined(HAVE_AVX2) && !defined(__AVX2__)
static_assert(HAVE_AVX2 == 0, "__AVX2__ should be defined when HAVE_AVX2 is "
                              "defined");
#endif

template <std::size_t NR_SAMPLES> struct LookupHighwayBackend<NR_SAMPLES>::Impl {
  std::vector<float> lookup;
  static constexpr std::size_t MASK = NR_SAMPLES - 1;
  static constexpr float SCALE = NR_SAMPLES / (2.0f * float(M_PI));

  void init() {
    lookup.resize(NR_SAMPLES);
    for (std::size_t i = 0; i < NR_SAMPLES; ++i) {
      lookup[i] = sinf(i * (2.0f * float(M_PI) / NR_SAMPLES));
    }
  }

  void compute_sincosf(std::size_t n, const float *x, float *s,
                       float *c) const {

  }

  void compute_sinf(std::size_t n, const float *x, float *s) const {
    highway_impl::compute_sinf(n, x, this->lookup.data(), this->MASK, this->SCALE, s);
  }

  void compute_cosf(std::size_t n, const float *x, float *c) const {
  }
};

template <std::size_t NR_SAMPLES>
LookupHighwayBackend<NR_SAMPLES>::LookupHighwayBackend()
    : impl(std::make_unique<Impl>()) {}

template <std::size_t NR_SAMPLES>
LookupHighwayBackend<NR_SAMPLES>::~LookupHighwayBackend() = default;

template <std::size_t NR_SAMPLES>
void LookupHighwayBackend<NR_SAMPLES>::init(size_t) {
  impl->init();
}

template <std::size_t NR_SAMPLES>
void LookupHighwayBackend<NR_SAMPLES>::compute_sinf(std::size_t n, const float *x,
                                                float *s) const {
  impl->compute_sinf(n, x, s);
}

template <std::size_t NR_SAMPLES>
void LookupHighwayBackend<NR_SAMPLES>::compute_cosf(std::size_t n, const float *x,
                                                float *c) const {
  impl->compute_cosf(n, x, c);
}

template <std::size_t NR_SAMPLES>
void LookupHighwayBackend<NR_SAMPLES>::compute_sincosf(std::size_t n,
                                                   const float *x, float *s,
                                                   float *c) const {
  impl->compute_sincosf(n, x, s, c);
}

// Explicit instantiations
template class LookupHighwayBackend<16384>;
template class LookupHighwayBackend<32768>;
