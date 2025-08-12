#include <catch2/catch_test_macros.hpp>
#include <trigdx/gpu.hpp>

#include "test_utils.hpp"

TEST_CASE("sinf") { test_sinf<GPUBackend>(1e-6f); }

TEST_CASE("cosf") { test_cosf<GPUBackend>(1e-6f); }

TEST_CASE("sincosf") { test_sincosf<GPUBackend>(1e-6f); }
