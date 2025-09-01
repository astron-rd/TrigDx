#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/gpu.cuh"
#include "trigdx/gpu.hpp"

struct GPUBackend::Impl {

  ~Impl() {
    if (h_x) {
      cudaFreeHost(h_x);
    }
    if (h_s) {
      cudaFreeHost(h_s);
    }
    if (h_c) {
      cudaFreeHost(h_c);
    }
    if (h_e) {
      cudaFreeHost(h_e);
    }
    if (d_x) {
      cudaFree(d_x);
    }
    if (d_s) {
      cudaFree(d_s);
    }
    if (d_c) {
      cudaFree(d_c);
    }
    if (d_e) {
      cudaFree(d_e);
    }
  }

  void init(size_t n) {
    const size_t bytes = n * sizeof(float);
    cudaMallocHost(&h_x, bytes);
    cudaMallocHost(&h_s, bytes);
    cudaMallocHost(&h_c, bytes);
    cudaMallocHost(&h_e, bytes);
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_s, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_e, bytes);
  }

  void compute_sinf(size_t n, const float *x, float *s) const {
    const size_t bytes = n * sizeof(float);
    std::memcpy(h_x, x, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    launch_sinf_kernel(d_x, d_s, n);
    cudaMemcpy(h_s, d_s, bytes, cudaMemcpyDeviceToHost);
    std::memcpy(s, h_s, bytes);
  }

  void compute_cosf(size_t n, const float *x, float *c) const {
    const size_t bytes = n * sizeof(float);
    std::memcpy(h_x, x, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    launch_cosf_kernel(d_x, d_c, n);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    std::memcpy(c, h_c, bytes);
  }

  void compute_sincosf(size_t n, const float *x, float *s, float *c) const {
    const size_t bytes = n * sizeof(float);
    std::memcpy(h_x, x, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    launch_sincosf_kernel(d_x, d_s, d_c, n);
    cudaMemcpy(h_s, d_s, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    std::memcpy(s, h_s, bytes);
    std::memcpy(c, h_c, bytes);
  }

  void compute_expf(size_t n, const float *x, float *e) const {
    const size_t bytes = n * sizeof(float);
    std::memcpy(h_x, x, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    launch_expf_kernel(d_x, d_e, n);
    cudaMemcpy(h_e, d_e, bytes, cudaMemcpyDeviceToHost);
    std::memcpy(e, h_e, bytes);
  }

  float *h_x = nullptr;
  float *h_s = nullptr;
  float *h_c = nullptr;
  float *h_e = nullptr;
  float *d_x = nullptr;
  float *d_s = nullptr;
  float *d_c = nullptr;
  float *d_e = nullptr;
};

GPUBackend::GPUBackend() : impl(std::make_unique<Impl>()) {}

GPUBackend::~GPUBackend() = default;

void GPUBackend::init(size_t n) { impl->init(n); }

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

void GPUBackend::compute_expf(size_t n, const float *x, float *e) const {
  impl->compute_expf(n, x, e);
}
