#pragma once

#include <cstddef>
#include <memory>

#include "interface.hpp"

template <size_t NR_SAMPLES> class LookupBackend : public Backend {
public:
  LookupBackend();
  ~LookupBackend() override;

  void init() override;
  void compute_sinf(size_t n, const float *x, float *s) const override;
  void compute_cosf(size_t n, const float *x, float *c) const override;
  void compute_sincosf(size_t n, const float *x, float *s,
                       float *c) const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
