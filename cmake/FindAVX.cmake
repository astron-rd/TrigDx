include(CheckCXXSourceRuns)

set(CMAKE_REQUIRED_FLAGS "-mavx") # for GCC/Clang; use /arch:AVX for MSVC

check_cxx_source_runs(
  "
#include <immintrin.h>
int main() {
  __m256 a = _mm256_set_ps(-1.0f,2.0f,-3.0f,4.0f,-1.0f,2.0f,-3.0f,4.0f);
  __m256 b = _mm256_set_ps(1.0f,2.0f,3.0f,4.0f,1.0f,2.0f,3.0f,4.0f);
  __m256 result = _mm256_add_ps(a,b);
  return 0;
}"
  HAVE_AVX)

message(STATUS "AVX support: " ${HAVE_AVX})
