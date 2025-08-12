#include <trigdx/trigdx.hpp>

#include "benchmark_utils.hpp"

int main() {
  benchmark_sinf<MKLBackend>();
  benchmark_cosf<MKLBackend>();
  benchmark_sincosf<MKLBackend>();
}
