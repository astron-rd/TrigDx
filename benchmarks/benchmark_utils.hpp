#pragma once

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

void init_x(float *x, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    x[i] = (i % 360) * 0.0174533f; // degrees to radians
  }
}

template <typename Backend>
static void benchmark_sinf(benchmark::State &state) {
  const size_t N = static_cast<size_t>(state.range(0));

  Backend backend;

  auto start = std::chrono::high_resolution_clock::now();
  backend.init(N);
  float *x =
      reinterpret_cast<float *>(backend.allocate_memory(N * sizeof(float)));
  float *s =
      reinterpret_cast<float *>(backend.allocate_memory(N * sizeof(float)));
  auto end = std::chrono::high_resolution_clock::now();
  state.counters["init_ms"] =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1.e3;

  init_x(x, N);

  for (auto _ : state) {
    backend.compute_sinf(N, x, s);
    benchmark::DoNotOptimize(s);
  }

  backend.free_memory(x);
  backend.free_memory(s);

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(N));
}

template <typename Backend>
static void benchmark_cosf(benchmark::State &state) {
  const size_t N = static_cast<size_t>(state.range(0));

  Backend backend;

  auto start = std::chrono::high_resolution_clock::now();
  backend.init(N);
  float *x =
      reinterpret_cast<float *>(backend.allocate_memory(N * sizeof(float)));
  float *c =
      reinterpret_cast<float *>(backend.allocate_memory(N * sizeof(float)));

  if (!x || !c) {
    throw std::runtime_error("Buffer allocation failed");
  }
  auto end = std::chrono::high_resolution_clock::now();
  state.counters["init_ms"] =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1.e3;

  init_x(x, N);

  for (auto _ : state) {
    backend.compute_cosf(N, x, c);
    benchmark::DoNotOptimize(c);
  }

  backend.free_memory(x);
  backend.free_memory(c);

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(N));
}

template <typename Backend>
static void benchmark_sincosf(benchmark::State &state) {
  const size_t N = static_cast<size_t>(state.range(0));

  Backend backend;

  auto start = std::chrono::high_resolution_clock::now();
  backend.init(N);
  float *x =
      reinterpret_cast<float *>(backend.allocate_memory(N * sizeof(float)));
  float *s =
      reinterpret_cast<float *>(backend.allocate_memory(N * sizeof(float)));
  float *c =
      reinterpret_cast<float *>(backend.allocate_memory(N * sizeof(float)));
  if (!x || !s || !c) {
    throw std::runtime_error("Buffer allocation failed");
  }
  auto end = std::chrono::high_resolution_clock::now();
  state.counters["init_ms"] =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1.e3;

  init_x(x, N);

  for (auto _ : state) {
    backend.compute_sincosf(N, x, s, c);
    benchmark::DoNotOptimize(s);
    benchmark::DoNotOptimize(c);
  }

  backend.free_memory(x);
  backend.free_memory(s);
  backend.free_memory(c);

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(N));
}
