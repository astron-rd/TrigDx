#pragma once

#include <cmath>

template <size_t NR_SAMPLES>
void LookupBackend<NR_SAMPLES>::init() {
    lookup.resize(NR_SAMPLES);
    for (size_t i = 0; i < NR_SAMPLES; ++i)
        lookup[i] = std::sinf(i * (2.0f * float(M_PI) / NR_SAMPLES));
}

template <size_t NR_SAMPLES>
void LookupBackend<NR_SAMPLES>::compute_sinf(size_t n,
                                               const float* x,
                                               float* s) const {
    for (size_t i = 0; i < n; ++i) {
        size_t idx = static_cast<size_t>(x[i] * SCALE) & MASK;
        s[i] = lookup[idx];
    }
}

template <size_t NR_SAMPLES>
void LookupBackend<NR_SAMPLES>::compute_cosf(size_t n,
                                               const float* x,
                                               float* c) const {
    for (size_t i = 0; i < n; ++i) {
        size_t idx = static_cast<size_t>(x[i] * SCALE) & MASK;
        size_t idx_cos = (idx + NR_SAMPLES / 4) & MASK;
        c[i] = lookup[idx_cos];
    }
}

template <size_t NR_SAMPLES>
void LookupBackend<NR_SAMPLES>::compute_sincosf(size_t n,
                                                  const float* x,
                                                  float* s,
                                                  float* c) const {
    for (size_t i = 0; i < n; ++i) {
        size_t idx = static_cast<size_t>(x[i] * SCALE) & MASK;
        size_t idx_cos = (idx + NR_SAMPLES / 4) & MASK;
        s[i] = lookup[idx];
        c[i] = lookup[idx_cos];
    }
}
