from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="onebit_sparse_mul",
    version="0.0.1",
    author="adeshen",
    author_email="roberto.lopez.castro@udc.es",
    description="Highly optimized FP16x(INT4+2:4 sparsity) CUDA matmul kernel.",
    install_requires=["numpy", "torch", "transformers"],
    packages=["onebit_sparse_mul"],
    ext_modules=[
        cpp_extension.CUDAExtension(
            "onebit_sparse_cuda",
            [
                "onebit_sparse_mul/torch_kernel_cuda.cpp",
            ],
            include_dirs=[
                "/root/OneBitQuantizer/OneBitSparseMul/include",
            ],
            
            library_dirs=[
                "."
            ],
            libraries=[
                "onebitmul"
            ],
            extra_compile_args={
                "nvcc": ["-arch=sm_86", "--ptxas-options=-v", "-lineinfo"
                         ," -I/root/OneBitQuantizer/OneBitSparseMul/csrc", 
                         " --std=c++17", "-lonebitmul"
                         ]
            },
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)