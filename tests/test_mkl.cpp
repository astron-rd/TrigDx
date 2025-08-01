#include <catch2/catch_test_macros.hpp>
#include <trigdx/mkl.hpp>

#include "test_utils.hpp"

TEST_CASE("sinf") { test_sinf<MKLBackend>(1e-6f); }

TEST_CASE("cosf") { test_cosf<MKLBackend>(1e-6f); }

TEST_CASE("sincosf") { test_sincosf<MKLBackend>(1e-6f); }
