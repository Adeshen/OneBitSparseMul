import onebit_sparse_cuda



def mul(A, B, C, meta):
    
    onebit_sparse_cuda.onebit_mul(A, B, C, meta)