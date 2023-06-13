#include <ATen/ATen.h>

at::Tensor find_intersections(
    at::Tensor vertices
    , at::Tensor faces
    , at::Tensor ray_origins
    , at::Tensor ray_directions
);

at::Tensor find_distances(
    at::Tensor vertices
    , at::Tensor faces
    , at::Tensor ray_origins
    , at::Tensor ray_directions
);

