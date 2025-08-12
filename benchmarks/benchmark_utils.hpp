#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

// Default values if not overridden by range multipliers
constexpr size_t DEFAULT_N = 10'000'000;

template <typename Backend>
static void benchmark_sinf(benchmark::State &state) {
  const size_t N = static_cast<size_t>(state.range(0));
  std::vector<float> x(N), s(N);

  for (size_t i = 0; i < N; ++i) {
    x[i] = (i % 360) * 0.0174533f; // degrees to radians
  }

  Backend backend;

  auto start = std::chrono::high_resolution_clock::now();
  backend.init(N);
  auto end = std::chrono::high_resolution_clock::now();
  state.counters["init_ms"] =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1.e3;

  for (auto _ : state) {
    backend.compute_sinf(N, x.data(), s.data());
    benchmark::DoNotOptimize(s);
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(N));
}

template <typename Backend>
static void benchmark_cosf(benchmark::State &state) {
  const size_t N = static_cast<size_t>(state.range(0));
  std::vector<float> x(N), c(N);

  for (size_t i = 0; i < N; ++i) {
    x[i] = (i % 360) * 0.0174533f;
  }

  Backend backend;

  auto start = std::chrono::high_resolution_clock::now();
  backend.init(N);
  auto end = std::chrono::high_resolution_clock::now();
  state.counters["init_ms"] =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1.e3;

  for (auto _ : state) {
    backend.compute_cosf(N, x.data(), c.data());
    benchmark::DoNotOptimize(c);
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(N));
}

template <typename Backend>
static void benchmark_sincosf(benchmark::State &state) {
  const size_t N = static_cast<size_t>(state.range(0));
  std::vector<float> x(N), s(N), c(N);

  for (size_t i = 0; i < N; ++i) {
    x[i] = (i % 360) * 0.0174533f;
  }

  Backend backend;

  auto start = std::chrono::high_resolution_clock::now();
  backend.init(N);
  auto end = std::chrono::high_resolution_clock::now();
  state.counters["init_ms"] =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1.e3;

  for (auto _ : state) {
    backend.compute_sincosf(N, x.data(), s.data(), c.data());
    benchmark::DoNotOptimize(s);
    benchmark::DoNotOptimize(c);
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(N));
}
