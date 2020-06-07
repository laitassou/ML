cmake_minimum_required(VERSION 3.10)
project(mpg)


set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
#include_directories(../../include)
add_executable(mpg  main.cpp ../../src/Model.cpp ../../src/Tensor.cpp ../../src/ModelInfos.cpp ../../src/LoaderHarness.cpp ../../src/BasicModelManager.cpp  )
target_include_directories(mpg PRIVATE ../../include)
target_link_libraries (mpg -ltensorflow)