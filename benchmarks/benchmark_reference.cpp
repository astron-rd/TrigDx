#include <trigdx/reference.hpp>

#include "benchmark_utils.hpp"

BENCHMARK_TEMPLATE(benchmark_sinf, ReferenceBackend)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1e5)
    ->Arg(1e6)
    ->Arg(1e7);
BENCHMARK_TEMPLATE(benchmark_cosf, ReferenceBackend)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1e5)
    ->Arg(1e6)
    ->Arg(1e7);
BENCHMARK_TEMPLATE(benchmark_sincosf, ReferenceBackend)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1e5)
    ->Arg(1e6)
    ->Arg(1e7);

BENCHMARK_MAIN();