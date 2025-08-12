#include <trigdx/gpu.hpp>

#include "benchmark_utils.hpp"

BENCHMARK_TEMPLATE(benchmark_sinf, GPUBackend)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1e5)
    ->Arg(1e6)
    ->Arg(1e7);
BENCHMARK_TEMPLATE(benchmark_cosf, GPUBackend)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1e5)
    ->Arg(1e6)
    ->Arg(1e7);
BENCHMARK_TEMPLATE(benchmark_sincosf, GPUBackend)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1e5)
    ->Arg(1e6)
    ->Arg(1e7);

BENCHMARK_MAIN();