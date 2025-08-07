#include <trigdx/lookup_xsimd.hpp>

#include "benchmark_utils.hpp"

int main() {
  benchmark_sinf<LookupXSIMDBackend<16384>>();
  benchmark_cosf<LookupXSIMDBackend<16384>>();
  benchmark_sincosf<LookupXSIMDBackend<16384>>();

  benchmark_sinf<LookupXSIMDBackend<32768>>();
  benchmark_cosf<LookupXSIMDBackend<32768>>();
  benchmark_sincosf<LookupXSIMDBackend<32768>>();
}
