#include <cuda_runtime.h>

#include "gpu.cuh"

__global__ void kernel_sinf(const float *__restrict__ x, float *__restrict__ s,
                            size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // s[idx] = __sinf(x[idx]);
    s[idx] = sinf(x[idx]);
  }
}

__global__ void kernel_cosf(const float *__restrict__ x, float *__restrict__ c,
                            size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // c[idx] = __cosf(x[idx]);
    c[idx] = cosf(x[idx]);
  }
}

__global__ void kernel_sincosf(const float *__restrict__ x,
                               float *__restrict__ s, float *__restrict__ c,
                               size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // __sincosf(x[idx], &s[idx], &c[idx]);
    s[idx] = sinf(x[idx]);
    c[idx] = cosf(x[idx]);
  }
}

__global__ void kernel_expf(const float *__restrict__ x, float *__restrict__ e,
                            size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // e[idx] = __expf(x[idx]);
    e[idx] = expf(x[idx]);
  }
}

namespace {
inline dim3 make_grid(size_t n, size_t threadsPerBlock = 256) {
  return dim3((n + threadsPerBlock - 1) / threadsPerBlock);
}
} // namespace

void launch_sinf_kernel(const float *d_x, float *d_s, size_t n) {
  dim3 blocks(256);
  dim3 grid = make_grid(n, blocks.x);
  kernel_sinf<<<grid, blocks>>>(d_x, d_s, n);
}

void launch_cosf_kernel(const float *d_x, float *d_c, size_t n) {
  dim3 blocks(256);
  dim3 grid = make_grid(n, blocks.x);
  kernel_cosf<<<grid, blocks>>>(d_x, d_c, n);
}

void launch_sincosf_kernel(const float *d_x, float *d_s, float *d_c, size_t n) {
  dim3 blocks(256);
  dim3 grid = make_grid(n, blocks.x);
  kernel_sincosf<<<grid, blocks>>>(d_x, d_s, d_c, n);
}

void launch_expf_kernel(const float *d_x, float *d_e, size_t n) {
  dim3 blocks(256);
  dim3 grid = make_grid(n, blocks.x);
  kernel_expf<<<grid, blocks>>>(d_x, d_e, n);
}
