#include <ATen/ATen.h>
#include <cuda.h>
#include "raycast_cuda.cuh"


at::Tensor measure_distance(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor ray_origins,
    at::Tensor ray_directions
) {
    return measure_distance_cuda(vertices, faces, ray_origins, ray_directions);
}