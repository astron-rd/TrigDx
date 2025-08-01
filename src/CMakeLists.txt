add_library(trigdx reference.cpp lookup.cpp)

target_include_directories(trigdx PUBLIC ${PROJECT_SOURCE_DIR}/include)

if(USE_MKL)
  find_package(MKL REQUIRED)
  target_sources(trigdx PRIVATE mkl.cpp)
  target_link_libraries(trigdx PRIVATE MKL::MKL)
endif()
