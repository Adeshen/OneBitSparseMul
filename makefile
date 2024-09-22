ROOT=.
CUTLASS_PATH=$(ROOT)/thirdparty/cutlass

cutlass_flag=-I$(CUTLASS_PATH)/include \
    -I$(CUTLASS_PATH)/tools/util/include \
	-I$(CUTLASS_PATH)/test \
	-I$(ROOT)/csrc\
	-L/usr/local/cuda/lib64 -lcuda  -lcudadevrt -lcudart_static -lcublas -arch=sm_80 \

NVCC=/usr/local/cuda/bin/nvcc

kernel:
	$(NVCC) csrc/kernel.cu $(cutlass_flag)