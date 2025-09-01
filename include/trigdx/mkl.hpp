#pragma once

#include "interface.hpp"

class MKLBackend : public Backend {
public:
  void compute_sinf(size_t n, const float *x, float *s) const override;

  void compute_cosf(size_t n, const float *x, float *c) const override;

  void compute_sincosf(size_t n, const float *x, float *s,
                       float *c) const override;

  void compute_expf(size_t n, const float *x, float *e) const override;
};
