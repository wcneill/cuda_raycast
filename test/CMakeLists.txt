add_executable(cuda_test test.cpp)
target_link_libraries(cuda_test libray)
target_include_directories(cuda_test PRIVATE "${CMAKE_SOURCE_DIR}/libray/include")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
find_package(Torch REQUIRED)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET cuda_test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:cuda_test>)
endif (MSVC)