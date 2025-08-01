#include <trigdx/reference.hpp>

#include "benchmark_utils.hpp"

int main() {
  benchmark_sinf<ReferenceBackend>();
  benchmark_cosf<ReferenceBackend>();
  benchmark_sincosf<ReferenceBackend>();
}
