#pragma once
#include <array>
#include <cmath>
#include <cstddef>

// Since C++17 does not yet have constexpr math functions, we define a simple sine approximation.
// In C++26, we could use constexpr std::sin directly.
constexpr float sin_approx(float x) {
    constexpr float PI = float(M_PI);
    constexpr float TWO_PI = 2.0f * PI;
    while (x > PI)  x -= TWO_PI;
    while (x < -PI) x += TWO_PI;

    float x2 = x * x;
    return x * (1 - x2 / 6 + x2 * x2 / 120);
}

template <std::size_t N>
struct lookup_constexpr {
    static_assert((N & (N - 1)) == 0, "N must be a power of two");

    static constexpr std::size_t MASK = N - 1;
    static constexpr float SCALE = N / (2.0f * float(M_PI));

    static constexpr std::array<float, N> generate() {
        std::array<float, N> arr{};
        for (std::size_t i = 0; i < N; ++i) {
            arr[i] = sin_approx(i * (2.0f * float(M_PI) / N));
        }
        return arr;
    }

    static constexpr std::array<float, N> values = generate();
};

template <std::size_t N>
constexpr std::array<float, N> lookup_constexpr<N>::values;


template <std::size_t NR_SAMPLES> class LookupCompileTimeBackend : public Backend {
public:
  LookupCompileTimeBackend();
  ~LookupCompileTimeBackend() override;

  void init(size_t n = 0) override;
  void compute_sinf(std::size_t n, const float *x, float *s) const override;
  void compute_cosf(std::size_t n, const float *x, float *c) const override;
  void compute_sincosf(std::size_t n, const float *x, float *s,
                       float *c) const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
