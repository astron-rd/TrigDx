#include <catch2/catch_test_macros.hpp>
#include <trigdx/lookup.hpp>

#include "test_utils.hpp"

TEST_CASE("sinf") {
  test_sinf<LookupBackend<16384>>(1e-2f);
  test_sinf<LookupBackend<32768>>(1e-2f);
}

TEST_CASE("cosf") {
  test_cosf<LookupBackend<16384>>(1e-2f);
  test_cosf<LookupBackend<32768>>(1e-2f);
}

TEST_CASE("sincosf") {
  test_sincosf<LookupBackend<16384>>(1e-2f);
  test_sincosf<LookupBackend<32768>>(1e-2f);
}