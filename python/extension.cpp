#include "raycast.h"
#include <torch/extension.h>

PYBIND11_MODULE(raycast, m) {
  m.def("find_distances", &find_distances, "Get minimum distance from ray origin to surface.");
  m.def("find_intersections", &find_intersections, "Get ray's intersection with surface.");
}