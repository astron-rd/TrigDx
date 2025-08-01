add_executable(benchmark_reference benchmark_reference.cpp)
target_link_libraries(benchmark_reference PRIVATE trigdx)

add_executable(benchmark_lookup benchmark_lookup.cpp)
target_link_libraries(benchmark_lookup PRIVATE trigdx)

if(USE_MKL)
  add_executable(benchmark_mkl benchmark_mkl.cpp)
  target_link_libraries(benchmark_mkl PRIVATE trigdx)
endif()
