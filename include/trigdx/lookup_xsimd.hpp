#pragma once

#include <cstddef>
#include <memory>

#include "interface.hpp"

template <std::size_t NR_SAMPLES> class LookupXSIMDBackend : public Backend {
public:
  LookupXSIMDBackend();
  ~LookupXSIMDBackend() override;

  void init(size_t n = 0) override;
  void compute_sinf(std::size_t n, const float *x, float *s) const override;
  void compute_cosf(std::size_t n, const float *x, float *c) const override;
  void compute_sincosf(std::size_t n, const float *x, float *s,
                       float *c) const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
