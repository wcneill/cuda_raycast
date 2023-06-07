
#include <stdexcept>
#include <ATen/ATen.h>
#include <torch/types.h>

#include <initializer_list>

void check_cuda(at::Tensor x);
void check_contiguous(at::Tensor x);
void run_checks(at::Tensor x);
void run_checks(std::initializer_list<at::Tensor> tensors);