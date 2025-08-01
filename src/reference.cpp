#include <cmath>

#include "trigdx/reference.hpp"

void ReferenceBackend::compute_sinf(size_t n, const float *x, float *s) const {
  for (size_t i = 0; i < n; ++i) {
    s[i] = std::sinf(x[i]);
  }
}

void ReferenceBackend::compute_cosf(size_t n, const float *x, float *c) const {
  for (size_t i = 0; i < n; ++i) {
    c[i] = std::cosf(x[i]);
  }
}

void ReferenceBackend::compute_sincosf(size_t n, const float *x, float *s,
                                       float *c) const {
  for (size_t i = 0; i < n; ++i) {
    s[i] = std::sinf(x[i]);
    c[i] = std::cosf(x[i]);
  }
}
