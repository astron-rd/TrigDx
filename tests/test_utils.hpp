#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trigdx/reference.hpp>

const size_t N = 1e7;

void init_x(std::vector<float> &x) {
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = (i % 360) * 0.0174533f; // degrees to radians
  }
}

template <typename Backend> inline void test_sinf(float tol) {
  std::vector<float> x(N), s_ref(N), s(N);
  init_x(x);

  ReferenceBackend ref;
  Backend backend;
  backend.init(N);

  ref.compute_sinf(N, x.data(), s_ref.data());
  backend.compute_sinf(N, x.data(), s.data());

  for (size_t i = 0; i < N; ++i) {
    REQUIRE_THAT(s[i], Catch::Matchers::WithinAbs(s_ref[i], tol));
  }
}

template <typename Backend> inline void test_cosf(float tol) {
  std::vector<float> x(N), c_ref(N), c(N);
  init_x(x);

  ReferenceBackend ref;
  Backend backend;
  backend.init(N);

  ref.compute_cosf(N, x.data(), c_ref.data());
  backend.compute_cosf(N, x.data(), c.data());

  for (size_t i = 0; i < N; ++i) {
    REQUIRE_THAT(c[i], Catch::Matchers::WithinAbs(c_ref[i], tol));
  }
}

template <typename Backend> inline void test_sincosf(float tol) {
  std::vector<float> x(N), s_ref(N), c_ref(N), s(N), c(N);
  init_x(x);

  ReferenceBackend ref;
  Backend backend;
  backend.init(N);

  ref.compute_sincosf(N, x.data(), s_ref.data(), c_ref.data());
  backend.compute_sincosf(N, x.data(), s.data(), c.data());

  for (size_t i = 0; i < N; ++i) {
    REQUIRE_THAT(s[i], Catch::Matchers::WithinAbs(s_ref[i], tol));
    REQUIRE_THAT(c[i], Catch::Matchers::WithinAbs(c_ref[i], tol));
  }
}
