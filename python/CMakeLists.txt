find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.0)
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(pytrigdx bindings.cpp)
target_link_libraries(pytrigdx PRIVATE trigdx)
