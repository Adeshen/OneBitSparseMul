import onebit_sparse_cuda

import torch
import torch.nn as nn

from onebit_sparse_mul._semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,
    mask_creator,
    sparse_semi_structured_to_dense_cutlass
)

def cuda_binary_mul(A, B, meta):
    C = torch.zeros((A.shape[0], B.shape[1]), dtype=torch.float32)
    onebit_sparse_cuda.onebit_mul(A, B, C, meta)
    return C

def cuda_binary_mul_(A, B, C, meta):
    onebit_sparse_cuda.onebit_mul(A, B, C, meta)
    return C


def binary_quant(tensor):
    return torch.where(tensor > 0, torch.tensor(1), torch.tensor(0))  # quant 

def pack_binary(tensor):
    shape = list(tensor.shape)
    shape[-1] = shape[-1] // 8
    
    packed_binary_tensor = torch.zeros(shape, dtype=torch.uint8)
    
    for i in range(8):
        packed_binary_tensor |= tensor[:, i::8] << i

    return packed_binary_tensor

def unpack_binary(packed_tensor):
    shape = list(packed_tensor.shape)
    shape[-1] = shape[-1] * 8

    unpacked_binary_tensor = torch.zeros(shape, dtype=torch.uint8)

    for i in range(shape[0]):
        for j in range(shape[1]):
            unpacked_binary_tensor[i, j] = (packed_tensor[i, j//8] >> (j%8) ) & 1
    return unpacked_binary_tensor

def dequant_desparse_tile(packed_sparse_tile, meta_tile):
    """
        1. unpack the packed_sparse_tile -> int32
        2. change it's 0 into -1
        3. sparse_semi_structured_to_dense_cutlass: transform it int32 from 2:4sparse into normal
    """
    k = packed_sparse_tile.shape[0]
    n = packed_sparse_tile.shape[1] * 8 

    # sparse_tile = torch.ones((k, n), dtype=torch.int16) 
    sparse_tile = unpack_binary(packed_sparse_tile)
    sparse_tile = torch.where(sparse_tile==1, torch.tensor(1), torch.tensor(-1))

    # print(packed_sparse_tile.shape)
    # print(sparse_tile.shape)
    # print(meta_tile.shape)
    dense_tile = sparse_semi_structured_to_dense_cutlass(sparse_tile, meta_tile)

    return dense_tile

def py_binary_gemm(x, packed_sparse_tensor, sparse_meta):
    m, k = x.shape
    n = packed_sparse_tensor.shape[1] * 8 * 2


    if packed_sparse_tensor.dtype != torch.uint8 :
        raise ValueError("packed_sparse_tensor must be `torch.uint8`")
    if packed_sparse_tensor.shape[0] != k:
        raise ValueError("x.shape[1] must equal to packed_sparse_tensor.shape[0], so we can matmul(x, packed_sparse_tensor)")
    if sparse_meta.dtype != torch.int16 :
        raise ValueError("Sparse Meta must be `torch.int16`")


    # out = torch.zeros((m, n), dtype=torch.float32)
    
    # thread_block_M = 16
    # thread_block_N = 16

    # for tile_m in range(0, m, thread_block_M):
    #     for tile_n in range(0, n, thread_block_N):
    #         tile_m_end =  tile_m + thread_block_M
    #         tile_n_end =  tile_n + thread_block_N

    #         dense_tile = dequant_desparse_tile(packed_sparse_tensor[:, tile_n//16:tile_n_end//16], 
    #                                          sparse_meta[:,tile_n//16:tile_n_end//16])

    #         out[tile_m:tile_m_end, tile_n:tile_n_end] = torch.matmul(x[tile_m:tile_m_end, :] , dense_tile)
    
    dense_tile = dequant_desparse_tile(packed_sparse_tensor, sparse_meta)
    out = torch.matmul(x, dense_tile.to(torch.float16))
    return out



class BinaryLinear_2_4(nn.Module):
    """PyTorch compatible Marlin layer; 1-bit 2:4 sparse linear layer without bias."""

    def __init__(self, 
                 infeatures: int, 
                 outfeatures: int
                 ):
        super().__init__()
        
        if infeatures % 128 != 0 or outfeatures != 256 == 0:
            raise ValueError(
                "`infeatures` must be divisible by 128 and `outfeatures` by 256."
            )
        self.packed_binary_sparse = torch.zeros( (infeatures, outfeatures//8//2), dtype=torch.uint8 )
        self.meta = torch.zeros( (infeatures, outfeatures//8), dtype=torch.int16 )

        self.vaild = False
    def forward(self, x):
        
        if not self.vaild :
            raise NotImplementedError("the BinaryLinear")
        
        # return py_binary_gemm(x, self.packed_binary_sparse, self.meta)
        sparse = self.packed_binary_sparse
        dense = x.T
        print(sparse.shape)
        print(dense.shape)
        return cuda_binary_mul(self.packed_binary_sparse, x,  self.meta)
    
    def pack(self, weight):

        mask = mask_creator(weight).to(torch.int32)
        w = weight * mask
        del mask

        sparse, self.meta = sparse_semi_structured_from_dense_cutlass(w)
        del w

        binary_sparse = binary_quant(sparse)
        del sparse

        binary_sparse = binary_sparse.T
        self.packed_binary_sparse = pack_binary(binary_sparse)
        # print(self.packed_binary_sparse.shape)
        self.vaild = True
