#include <trigdx/gpu.hpp>

#include "benchmark_utils.hpp"

int main() {
  benchmark_sinf<GPUBackend>();
  benchmark_cosf<GPUBackend>();
  benchmark_sincosf<GPUBackend>();
}
