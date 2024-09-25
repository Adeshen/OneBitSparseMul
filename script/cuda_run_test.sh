#!/bin/bash

# 定义不同的 a_rows 值
a_rows_values=(512 1024 2048 4096 8192 10240 12800) # 根据需要添加更多值

# 循环执行每个 a_rows 值
for a_rows in "${a_rows_values[@]}"; do
    echo "Running with a_rows=$a_rows"
    ./a.out --a_rows="$a_rows" --n="$a_rows" --a_cols="$a_rows" --reference-check=false --iterations=30
done