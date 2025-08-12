cmake_minimum_required(VERSION 3.16)
project(trigdx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TRIGDX_USE_MKL "Enable Intel MKL backend" OFF)
option(TRIGDX_USE_GPU "Enable GPU backend" OFF)
option(TRIGDX_USE_XSIMD "Enable XSIMD backend" OFF)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
