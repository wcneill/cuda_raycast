#include <iostream>
#include <ATen/ATen.h>
#include <torch/cuda.h>
#include "raycast_kernel.cu"


int main() {
	
	at::Tensor a = at::ones({ 2, 2 }, at::kInt);
	at::Tensor b = at::randn({ 2, 2 });
	auto c = a + b.to(at::kInt);

	std::cout << "A Tensor! " << c << std::endl;
	std::cout << "Cuda is availagle: " << torch::cuda::is_available() << std::endl;

};