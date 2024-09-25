import torch
import onebit_sparse_mul

from onebit_sparse_mul._semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,
    mask_creator,
    sparse_semi_structured_to_dense_cutlass,
)

# shape = (128, 128)
# shape = (1024, 1024)
# shape = (4096, 4096)
# shape = (8192, 8192)
shape = (8192*2, 8192*2)
w = torch.ones(shape, dtype = torch.int32) 
mask = mask_creator(w).to(torch.int32)
w = w * mask 
        
sparse, meta = sparse_semi_structured_from_dense_cutlass(w)

print(w[:2,:4], "w", w.shape)  # 0, 0, 1, 1
print(mask[:2,:4], "mask", mask.shape) # 0, 0, 1, 1
print(sparse[:2,:2], "sparse", sparse.shape) # 1, 1
print(meta[:1], "meta", meta.shape, meta.dtype) #  -4370 = 1110, 1110, 1110, 1110

packed_sparse = onebit_sparse_mul.pack_binary(sparse)
a = torch.ones(shape, dtype=torch.float16)


import time
c = torch.empty((packed_sparse.shape[0], a.shape[1]), dtype=torch.float32)

iters = 20

st = time.time()
for i in range(iters):
    onebit_sparse_mul.cuda_binary_mul_(packed_sparse, a, c, meta)
    # c = onebit_sparse_mul.cuda_binary_mul(packed_sparse, a, meta)
print("cost time", (time.time()-st)/iters)

print(c, "c ", c.shape)