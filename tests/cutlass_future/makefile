ROOT=../../
CUTLASS_PATH=../../thirdparty/cutlass

cutlass_flag=-I$(CUTLASS_PATH)/include \
    -I$(CUTLASS_PATH)/tools/util/include \
	-I$(CUTLASS_PATH)/test \
	-I$(ROOT)/build/_deps/googletest-src/googletest/include \
	-I$(ROOT)/tests/unit\
	-I$(ROOT)/csrc\
	-L/usr/local/cuda/lib64 -lcuda  -lcudadevrt -lcudart_static -lcublas -arch=sm_80 \
	-L$(ROOT)/build/lib

NVCC=/usr/local/cuda/bin/nvcc


array:
	$(NVCC) alignedarray_test.cu  $(cutlass_flag)  -lgtest -g
