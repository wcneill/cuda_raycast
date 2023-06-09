#include "raycast.h"
#include "raycast_cuda.cuh"


at::Tensor find_intersections(
    at::Tensor vertices
    , at::Tensor faces
    , at::Tensor ray_origins
    , at::Tensor ray_directions
) {
    at::Tensor intersect_distance = get_distance(vertices, faces, ray_origins, ray_directions); // [R, 1]
    at::Tensor intersection_coordinates = intersect_distance * ray_directions + ray_origins;    // [R, 3 (x, y, z)]
    return intersection_coordinates;
}