#include <iostream>

#include <ATen/ATen.h>
#include <torch/types.h>

#include "raycast_cuda.cuh"

int main() {

	auto face_options = at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto vert_options = at::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

	torch::Tensor vertices = torch::tensor(
		{{-1, -1, 0},
		 {1, 1, 0},
		 {-1, 1, 0}}, vert_options
	);

	torch::Tensor faces =  torch::tensor({{0, 1, 2}}, face_options);
	torch::Tensor ray_origins = torch::tensor(
		{{-.5, .5, 1.},
		 {.5, -.5, 1.},
		 {-.75, .75, 2.2}}, vert_options);

	torch::Tensor ray_directions = torch::tensor({{0, 0, -1}}, vert_options)
		.expand({ray_origins.size(0), -1})
		.contiguous();

    at::Tensor distances = get_distance(vertices, faces, ray_origins, ray_directions);
	std::cout << distances << std::endl;
}