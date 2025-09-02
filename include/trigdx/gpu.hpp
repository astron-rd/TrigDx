
#pragma once

#include <cstddef>
#include <memory>

#include "interface.hpp"

class GPUBackend : public Backend {
public:
  GPUBackend();
  ~GPUBackend() override;

  void init(size_t n = 0) override;
  void *allocate_memory(size_t bytes) const override;
  void free_memory(void *ptr) const override;
  void compute_sinf(size_t n, const float *x, float *s) const override;
  void compute_cosf(size_t n, const float *x, float *c) const override;
  void compute_sincosf(size_t n, const float *x, float *s,
                       float *c) const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
