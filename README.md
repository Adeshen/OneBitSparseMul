# One Bit Sparse Matmul




# Usage

## test kernel

Firstly, pull the denpendence `cutlass`

```
git submodule init
git submodule update
```


Secondly, compile the benchmark test
```
make 
```

Thirdly, run the 
```
./a.out --a_rows=1024 --n=1024 --a_cols=1024 --reference-check=false
```
## test python mul

```
git submodule init
git submodule update
```

Secondly, compile the benchmark test
```
make lib
```

Thirdly, install the python package
```
pip install -e .
```

Fourthly, run the base mul
```
python3 /root/OneBitQuantizer/OneBitSparseMul/tests/python/test_mul.py
```

# Benchmark test

## gemm 256x64x128_128x32x128_16x8x32

```
        using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 128>;
        using WarpShape = cutlass::gemm::GemmShape<128, 32, 128>;
        using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
```
### test result on GPU3060ti  
| a_rows |   n   | a_cols |    GFLOPs    | Runtime (ms) |
|--------|-------|--------|--------------|---------------|
|  512   |  512  |  512   |  14838.3     |   0.0180907   |
| 1024   | 1024  | 1024   |  27413.8     |   0.078336    |
| 2048   | 2048  | 2048   |  46478.6     |   0.36963     |
| 4096   | 4096  | 4096   |  52101.1     |   2.63793     |
| 8192   | 8192  | 8192   |  67349.1     |   16.3256     |
| 10240  | 10240 | 10240  |  66994.7     |   32.0545     |
| 12800  | 12800 | 12800  |  68902.6     |   31.1669     |
| 16384  | 16384 | 16384  |  70834.7     |   124.178     |
| 25600  | 25600 | 25600  |  69982.5     |   479.469     |


