#include <catch2/catch_test_macros.hpp>
#include <trigdx/lookup_avx.hpp>

#include "test_utils.hpp"

TEST_CASE("sinf") {
  test_sinf<LookupAVXBackend<16384>>(1e-2f);
  test_sinf<LookupAVXBackend<32768>>(1e-2f);
}

TEST_CASE("cosf") {
  test_cosf<LookupAVXBackend<16384>>(1e-2f);
  test_cosf<LookupAVXBackend<32768>>(1e-2f);
}

TEST_CASE("sincosf") {
  test_sincosf<LookupAVXBackend<16384>>(1e-2f);
  test_sincosf<LookupAVXBackend<32768>>(1e-2f);
}