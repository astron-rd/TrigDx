#include <trigdx/trigdx.hpp>

#include "benchmark_utils.hpp"

int main() {
  benchmark_sinf<LookupBackend<16384>>();
  benchmark_cosf<LookupBackend<16384>>();
  benchmark_sincosf<LookupBackend<16384>>();

  benchmark_sinf<LookupBackend<32768>>();
  benchmark_cosf<LookupBackend<32768>>();
  benchmark_sincosf<LookupBackend<32768>>();
}
