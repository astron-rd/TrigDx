FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4)
set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

add_executable(benchmark_reference benchmark_reference.cpp)
target_link_libraries(benchmark_reference PRIVATE trigdx benchmark::benchmark)

add_executable(benchmark_lookup benchmark_lookup.cpp)
target_link_libraries(benchmark_lookup PRIVATE trigdx benchmark::benchmark)

if(HAVE_AVX)
  add_executable(benchmark_lookup_avx benchmark_lookup_avx.cpp)
  target_link_libraries(benchmark_lookup_avx PRIVATE trigdx
                                                     benchmark::benchmark)
endif()

if(TRIGDX_USE_MKL)
  add_executable(benchmark_mkl benchmark_mkl.cpp)
  target_link_libraries(benchmark_mkl PRIVATE trigdx benchmark::benchmark)
endif()

if(TRIGDX_USE_GPU)
  add_executable(benchmark_gpu benchmark_gpu.cpp)
  target_link_libraries(benchmark_gpu PRIVATE trigdx gpu benchmark::benchmark)
endif()

if(TRIGDX_USE_XSIMD)
  add_executable(benchmark_lookup_xsimd benchmark_lookup_xsimd.cpp)
  target_link_libraries(benchmark_lookup_xsimd PRIVATE trigdx
                                                       benchmark::benchmark)
endif()
