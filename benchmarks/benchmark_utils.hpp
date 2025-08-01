#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

const size_t N = 1e7;

inline void report(const std::string &name, double sec, double throughput) {
  std::ios state(nullptr);
  state.copyfmt(std::cout);
  std::cout << std::setw(7) << name << " -> ";
  std::cout << "time: ";
  std::cout << std::fixed << std::setprecision(3) << std::setfill('0');
  std::cout << sec << " s, ";
  std::cout << "throughput: " << throughput << " M elems/sec\n";
  std::cout.copyfmt(state);
}

template <typename Backend> inline void benchmark_sinf() {
  std::vector<float> x(N), s(N);

  for (size_t i = 0; i < N; ++i)
    x[i] = (i % 360) * 0.0174533f; // degrees to radians

  Backend backend;
  backend.init();

  auto start = std::chrono::high_resolution_clock::now();
  backend.compute_sinf(N, x.data(), s.data());
  auto end = std::chrono::high_resolution_clock::now();

  double sec = std::chrono::duration<double>(end - start).count();
  double throughput = N / sec / 1e6;

  report("sinf", sec, throughput);
}

template <typename Backend> inline void benchmark_cosf() {
  std::vector<float> x(N), c(N);

  for (size_t i = 0; i < N; ++i)
    x[i] = (i % 360) * 0.0174533f; // degrees to radians

  Backend backend;
  backend.init();

  auto start = std::chrono::high_resolution_clock::now();
  backend.compute_cosf(N, x.data(), c.data());
  auto end = std::chrono::high_resolution_clock::now();

  double sec = std::chrono::duration<double>(end - start).count();
  double throughput = N / sec / 1e6;

  report("cosf", sec, throughput);
}

template <typename Backend> inline void benchmark_sincosf() {
  std::vector<float> x(N), s(N), c(N);

  for (size_t i = 0; i < N; ++i)
    x[i] = (i % 360) * 0.0174533f; // degrees to radians

  Backend backend;
  backend.init();

  auto start = std::chrono::high_resolution_clock::now();
  backend.compute_sincosf(N, x.data(), s.data(), c.data());
  auto end = std::chrono::high_resolution_clock::now();

  double sec = std::chrono::duration<double>(end - start).count();
  double throughput = N / sec / 1e6;

  report("sincosf", sec, throughput);
}
