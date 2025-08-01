#pragma once

#include <cmath>
#include <vector>

#include "interface.hpp"

template <size_t NR_SAMPLES> class LookupBackend : public Backend {
public:
  void init() override;
  void compute_sinf(size_t n, const float *x, float *s) const override;
  void compute_cosf(size_t n, const float *x, float *c) const override;
  void compute_sincosf(size_t n, const float *x, float *s,
                       float *c) const override;

private:
  std::vector<float> lookup;
  static constexpr size_t MASK = NR_SAMPLES - 1;
  static constexpr float SCALE = NR_SAMPLES / (2.0f * float(M_PI));
};

#include "lookup.tpp" // include implementation
