#include <trigdx/trigdx.hpp>

#include "benchmark_utils.hpp"

template <typename Backend> void register_benchmarks() {
  BENCHMARK_TEMPLATE(benchmark_sinf, Backend)
      ->Unit(benchmark::kMillisecond)
      ->Arg(1e5)
      ->Arg(1e6)
      ->Arg(1e7);
  BENCHMARK_TEMPLATE(benchmark_cosf, Backend)
      ->Unit(benchmark::kMillisecond)
      ->Arg(1e5)
      ->Arg(1e6)
      ->Arg(1e7);
  BENCHMARK_TEMPLATE(benchmark_sincosf, Backend)
      ->Unit(benchmark::kMillisecond)
      ->Arg(1e5)
      ->Arg(1e6)
      ->Arg(1e7);
}

int main(int argc, char **argv) {
  ::benchmark::Initialize(&argc, argv);

  register_benchmarks<LookupBackend<16384>>();
  register_benchmarks<LookupBackend<32768>>();

  return ::benchmark::RunSpecifiedBenchmarks();
}