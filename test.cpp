#include <iostream>
#include <torch/types.h>
#include <torch/cuda.h>

#include "raycast.h"


int main() {

	auto face_options = at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto vert_options = at::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);

	torch::Tensor vertices = torch::tensor(
		{{-1, 1, 0},
		 {1, 1, 0},
		 {-1, -1, 0}}, vert_options
	);

	torch::Tensor faces =  torch::tensor({{2, 1, 0}}, face_options);
	torch::Tensor ray_origins = torch::tensor({{-.5, .5, 1.}}, vert_options);
	torch::Tensor ray_directions = torch::tensor({{0, 0, -1}}, vert_options);

	std::cout << vertices << std::endl;
	std::cout << faces << std::endl;
	std::cout << ray_origins << std::endl;
	std::cout << ray_directions << std::endl;

	at::Tensor output = measure_distance(vertices, faces, ray_origins, ray_directions);
	std::cout << output << std::endl;

}