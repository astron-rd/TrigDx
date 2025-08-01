include(FetchContent)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.9.0)
FetchContent_MakeAvailable(catch2)

# Lookup backend test
add_executable(test_lookup test_lookup.cpp)
target_link_libraries(test_lookup PRIVATE trigdx Catch2::Catch2WithMain)

# MKL backend test
if(USE_MKL)
  add_executable(test_mkl test_mkl.cpp)
  target_link_libraries(test_mkl PRIVATE trigdx Catch2::Catch2WithMain)
endif()

include(CTest)
add_test(NAME test_lookup COMMAND test_lookup)

if(USE_MKL)
  add_test(NAME test_mkl COMMAND test_mkl)
endif()
