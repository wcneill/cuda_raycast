#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

__device__ float_t* sub3d(float_t v1[3], float_t v2[3], float_t result[3]);
__device__ float_t* add3d(float_t v1[3], float_t v2[3], float_t result[3]);
__device__ float_t* scaler_mult3d(float_t vector[3], float_t scalar, float_t result[3]);
__device__ float_t dot3d(float_t v1[3], float_t v2[3]);
__device__ float_t* cross3d(float_t v1[3], float_t v2[3], float_t result[3]);

__global__ 
void distance_kernel(
    int n_rays, int n_faces
    , at::PackedTensorAccessor32<float_t, 2> vertex_acc
    , at::PackedTensorAccessor32<int32_t, 2> face_acc
    , at::PackedTensorAccessor32<float_t, 2> origin_acc
    , at::PackedTensorAccessor32<float_t, 2> direct_acc
    , at::PackedTensorAccessor32<float_t, 2> results
);

__host__
at::Tensor get_distance(
    at::Tensor vertices
    , at::Tensor faces
    , at::Tensor ray_origins
    , at::Tensor ray_directions
);