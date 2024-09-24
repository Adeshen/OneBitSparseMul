import torch
import onebit_sparse_mul

from onebit_sparse_mul._semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,
    mask_creator,
    sparse_semi_structured_to_dense_cutlass,
)

w = torch.ones((128, 32), dtype = torch.int32) 
mask = mask_creator(w).to(torch.int32)
w = w * mask 
        
sparse, meta = sparse_semi_structured_from_dense_cutlass(w)

print(w[:2,:4], "w", w.shape)  # 0, 0, 1, 1
print(mask[:2,:4], "mask", mask.shape) # 0, 0, 1, 1
print(sparse[:2,:2], "sparse", sparse.shape) # 1, 1
print(meta[:1], "meta", meta.shape, meta.dtype) #  -4370 = 1110, 1110, 1110, 1110


onebit_sparse_mul.mul()