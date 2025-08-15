#include <catch2/catch_test_macros.hpp>
#include <trigdx/lookup_xsimd.hpp>

#include "test_utils.hpp"

TEST_CASE("sincosf") {
  test_sincosf<LookupXSIMDBackend<16384>>(1e-6f);
  test_sincosf<LookupXSIMDBackend<32768>>(1e-6f);
}

TEST_CASE("sinf") {
  test_sinf<LookupXSIMDBackend<16384>>(1e-6f);
  test_sinf<LookupXSIMDBackend<32768>>(1e-6f);
}

TEST_CASE("cosf") {
  test_cosf<LookupXSIMDBackend<16384>>(1e-6f);
  test_cosf<LookupXSIMDBackend<32768>>(1e-6f);
}