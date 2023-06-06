#include <ATen/ATen.h>
#include <cuda.h>
#include <iostream>
#include "raycast_cuda.cuh"


at::Tensor measure_distance(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor ray_origins,
    at::Tensor ray_directions
) {
    std::cout << "I'm inside raycast.cpp - measure_distance" << std::endl;
    return measure_distance_cuda(vertices, faces, ray_origins, ray_directions);
}