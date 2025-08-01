cmake_minimum_required(VERSION 3.16)
project(trigdx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_MKL "Enable Intel MKL backend" OFF)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
