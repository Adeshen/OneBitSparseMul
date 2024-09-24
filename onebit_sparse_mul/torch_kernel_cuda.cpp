#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/python.h>

#include "onebit_sparse_tensor.h"



void onebit_mul(const torch::Tensor &A, const torch::Tensor &B,
             const torch::Tensor &meta, torch::Tensor &C
             ) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1) * 2;

  if (prob_n == B.size(1))
    AT_ERROR("AT error: C.size(1)!=B.size(1) ", prob_k);

  onebit_sparse_matmul(
    A.data_ptr(),
    B.data_ptr(),
    C.data_ptr(),
    C.data_ptr(),
    meta.data_ptr(),
    prob_m,
    prob_n,
    prob_k
  );
  // printf("groupsize is:%d\n", groupsize);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("onebit_mul", &onebit_mul, "A32 = Sparse W1 * A16 matmul.");
}