#include <ATen/ATen.h>

at::Tensor measure_distance(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor ray_origins,
    at::Tensor ray_directions
);