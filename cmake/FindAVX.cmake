include(CheckCXXSourceRuns)

set(SUPPORTED_COMPILERS Clang;GNU;Intel)

if(CMAKE_CXX_COMPILER_ID IN_LIST SUPPORTED_COMPILERS)
  set(CMAKE_REQUIRED_FLAGS "-mavx") # for GCC/Clang; use /arch:AVX for MSVC
else()
  message(FATAL_ERROR "Compiler : " ${CMAKE_CXX_COMPILER_ID} " not supported")
endif()

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

if(HAVE_AVX)
  message(STATUS "AVX support: true")
else()
  message(STATUS "AVX support: false")
endif()
