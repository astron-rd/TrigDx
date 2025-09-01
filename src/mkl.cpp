#include <mkl_vml.h>

#include "trigdx/mkl.hpp"

void MKLBackend::compute_sinf(size_t n, const float *x, float *s) const {
  vmsSin(static_cast<MKL_INT>(n), x, s, VML_HA);
}

void MKLBackend::compute_cosf(size_t n, const float *x, float *c) const {
  vmsCos(static_cast<MKL_INT>(n), x, c, VML_HA);
}

void MKLBackend::compute_sincosf(size_t n, const float *x, float *s,
                                 float *c) const {
  vmsSinCos(static_cast<MKL_INT>(n), x, s, c, VML_HA);
}

void MKLBackend::compute_expf(size_t n, const float *x, float *e) const {
  vmsExp(static_cast<MKL_INT>(n), x, e, VML_HA);
}
