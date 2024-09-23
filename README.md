# One Bit Sparse Matmul




# Usage

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
