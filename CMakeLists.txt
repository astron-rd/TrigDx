cmake_minimum_required(VERSION 3.16)
project(trigdx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TRIGDX_USE_MKL "Enable Intel MKL backend" OFF)
option(TRIGDX_USE_GPU "Enable GPU backend" OFF)
option(TRIGDX_USE_XSIMD "Enable XSIMD backend" OFF)
option(TRIGDX_BUILD_TESTS "Build tests" ON)
option(TRIGDX_BUILD_BENCHMARKS "Build tests" ON)
option(TRIGDX_BUILD_PYTHON "Build Python interface" ON)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/trigdx_config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/trigdx/trigdx_config.hpp @ONLY)

if(TRIGDX_BUILD_TESTS
   OR TRIGDX_BUILD_BENCHMARKS
   OR TRIGDX_BUILD_PYTHON)
  include(FetchContent)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)

if(TRIGDX_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(tests)
endif()

if(TRIGDX_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(TRIGDX_BUILD_PYTHON)
  add_subdirectory(python)
endif()
