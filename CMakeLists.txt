cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(raycast 
  VERSION 1.0
  DESCRIPTION "Cuda raycast experiment"
  LANGUAGES CXX CUDA)

add_executable(cuda_test 
  src/test.cpp src/raycast_cuda.cu src/torch_checks.cpp)

set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/includes/libtorch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/includes/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set_property(TARGET cuda_test PROPERTY CXX_STANDARD 14)

target_link_libraries(cuda_test "${TORCH_LIBRARIES}")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET cuda_test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:cuda_test>)
endif (MSVC)