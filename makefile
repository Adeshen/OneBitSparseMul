ROOT=.
CUTLASS_PATH=$(ROOT)/thirdparty/cutlass

cutlass_flag=-I$(CUTLASS_PATH)/include \
    -I$(CUTLASS_PATH)/tools/util/include \
	-I$(CUTLASS_PATH)/test \
	-I$(ROOT)/csrc\
	-I$(ROOT)/include\
	-L/usr/local/cuda/lib64 -lcuda  -lcudadevrt -lcudart_static -lcublas -arch=sm_80 \

NVCC=/usr/local/cuda/bin/nvcc

kernel:
	$(NVCC) csrc/kernel_lib.cu  tests/kernel_test.cu $(cutlass_flag)

lib:
	$(NVCC) csrc/kernel_lib.cu $(cutlass_flag) -shared --compiler-options -fPIC -o libonebitmul.so