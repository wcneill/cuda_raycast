cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(raycast 
  VERSION 1.0
  DESCRIPTION "Cuda raycast experiment"
  LANGUAGES CXX CUDA)

add_executable(cuda_test 
  test.cpp raycast.cpp raycast_cuda.cu)

set(Torch_DIR C:/Users/wesle/source/libtorch-win-shared-with-deps-2.0.1+cu118/libtorch/share/cmake/Torch)

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