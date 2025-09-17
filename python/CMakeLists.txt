find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.0)
  FetchContent_MakeAvailable(pybind11)
endif()

# Needed to set ${Python_VERSION_MAJOR} and ${Python_VERSION_MINOR}
find_package(
  Python REQUIRED
)

pybind11_add_module(pytrigdx bindings.cpp)
target_link_libraries(pytrigdx PRIVATE trigdx)
set_target_properties(pytrigdx PROPERTIES OUTPUT_NAME "trigdx")

set(PYTHON_SITE_PACKAGES
    "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/trigdx"
)

install(TARGETS pytrigdx DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES __init__.py DESTINATION ${PYTHON_SITE_PACKAGES})
