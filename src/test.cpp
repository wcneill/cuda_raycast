#include <iostream>

#include <ATen/ATen.h>
#include <torch/types.h>

#include "raycast_cuda.cuh"

template <typename T, size_t size> void print(const T (&array)[size])
{
	std::cout << "[";
    for(size_t i = 0; i < size; ++i)
        std::cout << array[i] << " ";
	std::cout << "]\n";
}

int main() {

	auto face_options = at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto vert_options = at::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

	torch::Tensor vertices = torch::tensor(
		{{-1, -1, 0},
		 {1, 1, 0},
		 {-1, 1, 0}}, vert_options
	);

	torch::Tensor faces =  torch::tensor({{0, 1, 2}}, face_options);
	torch::Tensor ray_origins = torch::tensor({{-.5, .5, 1.}}, vert_options);
	torch::Tensor ray_directions = torch::tensor({{0, 0, -1}}, vert_options);

	std::cout << vertices << std::endl;
    at::Tensor distances = measure_distance_cuda(vertices, faces, ray_origins, ray_directions);
	std::cout << distances << std::endl;
}