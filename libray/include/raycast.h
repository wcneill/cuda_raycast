#include <ATen/ATen.h>

at::Tensor find_intersections(
    at::Tensor vertices
    , at::Tensor faces
    , at::Tensor ray_origins
    , at::Tensor ray_directions
);