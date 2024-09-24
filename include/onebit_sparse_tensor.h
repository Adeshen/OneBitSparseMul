#pragma once
#include <cuda_runtime.h>


void onebit_sparse_matmul(void *a, void *b, void *c, void *d, void *e, int m, int n, int k);