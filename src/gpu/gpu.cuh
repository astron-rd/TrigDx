#pragma once

#include <cstddef>

void launch_sinf_kernel(const float *d_x, float *d_s, size_t n);
void launch_cosf_kernel(const float *d_x, float *d_c, size_t n);
void launch_sincosf_kernel(const float *d_x, float *d_s, float *d_c,
                           std::size_t n);
void launch_expf_kernel(const float *d_x, float *d_e, size_t n);
