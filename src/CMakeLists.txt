add_library(trigdx reference.cpp lookup.cpp lookup_avx.cpp)

target_include_directories(trigdx PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_compile_options(trigdx PRIVATE -O3 -march=native)

if(USE_MKL)
  find_package(MKL REQUIRED)
  target_sources(trigdx PRIVATE mkl.cpp)
  target_link_libraries(trigdx PRIVATE MKL::MKL)
endif()
