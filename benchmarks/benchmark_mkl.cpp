#include <trigdx/mkl.hpp>

#include "benchmark_utils.hpp"

BENCHMARK_TEMPLATE(benchmark_sinf, MKLBackend)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1e5)
    ->Arg(1e6)
    ->Arg(1e7);
BENCHMARK_TEMPLATE(benchmark_cosf, MKLBackend)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1e5)
    ->Arg(1e6)
    ->Arg(1e7);
BENCHMARK_TEMPLATE(benchmark_sincosf, MKLBackend)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1e5)
    ->Arg(1e6)
    ->Arg(1e7);

BENCHMARK_MAIN();