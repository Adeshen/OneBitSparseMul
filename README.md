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