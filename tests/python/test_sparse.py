import unittest

import torch
import onebit_sparse_mul

from onebit_sparse_mul._semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,
    mask_creator,
    sparse_semi_structured_to_dense_cutlass,
)

from onebit_sparse_mul import BinaryLinear_2_4, binary_quant, pack_binary, dequant_desparse_tile

class TestSparse(unittest.TestCase):
    def test_sparse_struct_Build(self):
        w = torch.ones((128, 32), dtype = torch.int32) 
        mask = mask_creator(w).to(torch.int32)
        w = w * mask 
               
        sparse, meta = sparse_semi_structured_from_dense_cutlass(w)

        print(w[:2,:4], "w", w.shape)  # 0, 0, 1, 1
        print(mask[:2,:4], "mask", mask.shape) # 0, 0, 1, 1
        print(sparse[:2,:2], "sparse", sparse.shape) # 1, 1
        print(meta[:1], "meta", meta.shape, meta.dtype) #  -4370 = 1110, 1110, 1110, 1110
        
    def test_quant_pack(self):
        w = torch.randint(-10, 10, (128,32), dtype=torch.int32)
        mask = mask_creator(w).to(torch.int32)
        w = w * mask
        sparse, meta = sparse_semi_structured_from_dense_cutlass(w)
    
        binary_sparse = binary_quant(sparse)
        packed_binary_sparse = pack_binary(binary_sparse)

        print(w[:2,:16], "w", w.shape)  
        print(mask[:2,:16], "mask", mask.shape)
        print(sparse[:2,:8], "sparse", sparse.shape) 
        print(binary_sparse[:2,:8], "binary_sparse", binary_sparse.shape) 
        print(packed_binary_sparse[:2,:1], "packed_binary_sparse", packed_binary_sparse.shape) 
        print(meta[:1], "meta", meta.shape, meta.dtype)

        linear = BinaryLinear_2_4(w.shape[0], w.shape[1])

        linear.pack(w)

        print(linear.packed_binary_sparse[:2,:1], "linear.packed_binary_sparse", linear.packed_binary_sparse.shape) 
        '''
        tensor([[ -7, -10,   0,   0,  -9, -10,   0,   0,   0,   0,   7,   2,   7, -10,
           0,   0],
        [  9,  -9,   0,   0,   0,   4,   0, -10,   0,   0,  -8,   7,   0,  -4,
           8,   0]], dtype=torch.int32) w torch.Size([128, 32])

        tensor([[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0]], dtype=torch.int32) mask torch.Size([128, 32])

        tensor([[ -7, -10,  -9, -10,   7,   2,   7, -10],
                [  9,  -9,   4, -10,  -8,   7,  -4,   8]], dtype=torch.int32) sparse torch.Size([128, 16])

        tensor([[0, 0, 0, 0, 1, 1, 1, 0],
                [1, 0, 1, 0, 0, 1, 0, 1]]) binary_sparse torch.Size([128, 16])

        tensor([[112],
                [165]], dtype=torch.uint8) packed_binary_sparse torch.Size([128, 2])

        tensor([[ 20036],   -> 0100 1110 0100 0100
                [-25138]], dtype=torch.int16) meta torch.Size([128, 2]) torch.int16
        '''
        pass


    def test_dequant(self):

        w = torch.ones((128, 32), dtype = torch.float16) 
        mask = mask_creator(w)
        w =( w * mask ).to(torch.float16)
               
        sparse, meta = sparse_semi_structured_from_dense_cutlass(w)

        # sparse = sparse.to(torch.float16)
        w_ = sparse_semi_structured_to_dense_cutlass(sparse, meta)

        linear = BinaryLinear_2_4(w.shape[0], w.shape[1])

        linear.pack(w)

        weight = dequant_desparse_tile(linear.packed_binary_sparse, linear.meta)

        print(w[:2,:4], "w", w.shape)  # 0, 0, 1, 1
        print(w_[:2,:4], "w_", w.shape)  # 0, 0, 1, 1
        print(weight[:2,:4], "weight", weight.shape, weight.dtype) 
        print(mask[:2,:4], "mask", mask.shape) # 0, 0, 1, 1
        print(sparse[:2,:2], "sparse", sparse.shape, sparse.dtype) # 1, 1
        print(meta[:1], "meta", meta.shape, meta.dtype) #  -4370 = 1110, 1110, 1110, 1110


    def test_sparse_gemmm(self):
        w = torch.ones((128, 32), dtype = torch.float16) 
        mask = mask_creator(w)
        w =( w * mask ).to(torch.float16)

        linear = BinaryLinear_2_4(w.shape[0], w.shape[1])

        linear.pack(w)
        print(linear.packed_binary_sparse[:2,:1], "linear.packed_binary_sparse", 
              linear.packed_binary_sparse.shape) 
        a = torch.ones((16,128), dtype=torch.float16)
        out = linear(a)

        print(out)

if __name__ == "__main__":
    unittest.main()