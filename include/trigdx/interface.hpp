#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

// Base interface for all math backends
class Backend {
public:
  virtual ~Backend() = default;

  // Optional initialization
  virtual void init(size_t n = 0) {}

  virtual void *allocate_memory(size_t bytes) const {
    return static_cast<void *>(new uint8_t[bytes]);
  };

  virtual void free_memory(void *ptr) const { std::free(ptr); };

  // Compute sine for n elements
  virtual void compute_sinf(size_t n, const float *x, float *s) const = 0;

  // Compute cosine for n elements
  virtual void compute_cosf(size_t n, const float *x, float *c) const = 0;

  // Compute sine and cosine for n elements
  virtual void compute_sincosf(size_t n, const float *x, float *s,
                               float *c) const = 0;
};
