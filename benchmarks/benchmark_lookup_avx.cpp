#include <trigdx/lookup_avx.hpp>

#include "benchmark_utils.hpp"

int main() {
  benchmark_sinf<LookupAVXBackend<16384>>();
  benchmark_cosf<LookupAVXBackend<16384>>();
  benchmark_sincosf<LookupAVXBackend<16384>>();

  benchmark_sinf<LookupAVXBackend<32768>>();
  benchmark_cosf<LookupAVXBackend<32768>>();
  benchmark_sincosf<LookupAVXBackend<32768>>();
}
