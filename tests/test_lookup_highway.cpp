#include <catch2/catch_test_macros.hpp>
#include <trigdx/lookup_highway.hpp>

#include "test_utils.hpp"

TEST_CASE("sincosf") {
  test_sincosf<LookupHighwayBackend<16384>>(1e-2f);
  test_sincosf<LookupHighwayBackend<32768>>(1e-2f);
}

TEST_CASE("sinf") {
  test_sinf<LookupHighwayBackend<16384>>(1e-2f);
  test_sinf<LookupHighwayBackend<32768>>(1e-2f);
}

TEST_CASE("cosf") {
  test_cosf<LookupHighwayBackend<16384>>(1e-2f);
  test_cosf<LookupHighwayBackend<32768>>(1e-2f);
}
