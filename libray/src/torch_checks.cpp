#include "torch_checks.h"
#include <ATen/ATen.h>
#include <c10/core/Device.h>


void check_cuda(at::Tensor x) {
    if (!x.device().is_cuda()) {
        throw std::invalid_argument("Expected all inputs to be on device: CUDA.");
    }
}

void check_contiguous(at::Tensor x) {
    if (!x.is_contiguous()) {
        throw std::invalid_argument("Expected all inputs to be contiguous.");
    }
}

void run_checks(at::Tensor x) {
    check_contiguous(x);
    check_cuda(x);
}

void run_checks(std::initializer_list<at::Tensor> tensors) {
    for (auto t : tensors) {
        run_checks(t);
    }
}

