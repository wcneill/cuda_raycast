cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(raycast 
  VERSION 1.0
  DESCRIPTION "Cuda raycast experiment"
  LANGUAGES CXX CUDA)

# set variables required to depend on libtorch, 
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/libtorch/share/cmake/Torch") 

add_subdirectory(libray)
add_subdirectory(test)