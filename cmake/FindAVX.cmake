include(CheckCXXSourceRuns)

set(SUPPORTED_COMPILERS Clang;GNU;Intel;IntelLLVM)

if(CMAKE_CXX_COMPILER_ID IN_LIST SUPPORTED_COMPILERS)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_REQUIRED_FLAGS "-xHost") # ICC
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_REQUIRED_FLAGS "-march=native") # ICX
  else()
    set(CMAKE_REQUIRED_FLAGS "-march=native") # GCC/Clang
  endif()
else()
  message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}.")
endif()

# AVX check
check_cxx_source_runs(
  "
#include <immintrin.h>
int main() {
    __m256 a = _mm256_setzero_ps(); // AVX
    (void) a;
    return 0;
}
"
  HAVE_AVX)

# AVX2 check
check_cxx_source_runs(
  "
#include <immintrin.h>
int main() {
    __m256i a = _mm256_set1_epi32(-1);
    __m256i b = _mm256_abs_epi32(a); // AVX2
    (void) b;
    return 0;
}
"
  HAVE_AVX2)
