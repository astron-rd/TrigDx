#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/gpu.cuh"
#include "trigdx/gpu.hpp"

struct GPUBackend::Impl {

  void *allocate_memory(size_t bytes) const {
    void *ptr;
    cudaMallocHost(&ptr, bytes);
    return ptr;
  }

  void free_memory(void *ptr) const { cudaFreeHost(ptr); }

  void compute_sinf(size_t n, const float *x, float *s) const {
    const size_t bytes = n * sizeof(float);
    float *d_x, *d_s;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_s, bytes);
    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    launch_sinf_kernel(d_x, d_s, n);
    cudaMemcpy(s, d_s, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_s);
  }

  void compute_cosf(size_t n, const float *x, float *c) const {
    const size_t bytes = n * sizeof(float);
    float *d_x, *d_c;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    launch_cosf_kernel(d_x, d_c, n);
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_c);
  }

  void compute_sincosf(size_t n, const float *x, float *s, float *c) const {
    const size_t bytes = n * sizeof(float);
    float *d_x, *d_s, *d_c;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_s, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    launch_sincosf_kernel(d_x, d_s, d_c, n);
    cudaMemcpy(s, d_s, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_s);
    cudaFree(d_c);
  }
};

GPUBackend::GPUBackend() : impl(std::make_unique<Impl>()) {}

GPUBackend::~GPUBackend() = default;

void *GPUBackend::allocate_memory(size_t bytes) const {
  return impl->allocate_memory(bytes);
}

void GPUBackend::free_memory(void *ptr) const { impl->free_memory(ptr); }

void GPUBackend::compute_sinf(size_t n, const float *x, float *s) const {
  impl->compute_sinf(n, x, s);
}

void GPUBackend::compute_cosf(size_t n, const float *x, float *c) const {
  impl->compute_cosf(n, x, c);
}

void GPUBackend::compute_sincosf(size_t n, const float *x, float *s,
                                 float *c) const {
  impl->compute_sincosf(n, x, s, c);
}