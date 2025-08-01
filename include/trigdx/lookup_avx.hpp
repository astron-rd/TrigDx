#pragma once

#include <cstddef>
#include <memory>

#include "interface.hpp"

template <std::size_t NR_SAMPLES> class LookupAVXBackend : public Backend {
public:
  LookupAVXBackend();
  ~LookupAVXBackend() override;

  void init() override;
  void compute_sinf(std::size_t n, const float *x, float *s) const override;
  void compute_cosf(std::size_t n, const float *x, float *c) const override;
  void compute_sincosf(std::size_t n, const float *x, float *s,
                       float *c) const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
