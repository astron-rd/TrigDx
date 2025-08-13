#pragma once

#include <trigdx/trigdx_config.hpp>

#include <trigdx/lookup.hpp>
#include <trigdx/lookup_avx.hpp>
#include <trigdx/reference.hpp>

#if defined(TRIGDX_USE_MKL)
#include <trigdx/mkl.hpp>
#endif

#if defined(TRIGDX_USE_GPU)
#include <trigdx/gpu.hpp>
#endif

#if defined(TRIGDX_USE_XSIMD)
#include <trigdx/lookup_xsimd.hpp>
#endif