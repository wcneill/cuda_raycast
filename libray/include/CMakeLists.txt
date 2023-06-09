set(UTILITY_DIR ${CMAKE_SOURCE_DIR}/libray/src)
set(API_DIR ${CMAKE_SOURCE_DIR}/libray/include)

add_library(libray 
    "${API_DIR}/raycast.cpp" 
    "${UTILITY_DIR}/raycast_cuda.cu"
    "${UTILITY_DIR}/torch_checks.cpp")

target_include_directories(libray 
    PUBLIC "${API_DIR}"
    PRIVATE "${UTILITY_DIR}"
)

find_package(Torch REQUIRED)
target_link_libraries(libray "${TORCH_LIBRARIES}")