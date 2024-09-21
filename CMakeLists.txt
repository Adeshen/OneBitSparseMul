cmake_minimum_required(VERSION 3.14)
project(sparse_kernel LANGUAGES CXX CUDA)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 编译器
find_package(CUDA REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/CUDA.cmake)
set(CMAKE_LANGUAGE_COMPILER )
# 设置 CUTLASS 路径
set(CUTLASS_PATH ./thirdparty/cutlass)
# 包含路径
include_directories(
    /usr/local/cuda/include
    ${CUTLASS_PATH}/include
    ${CUTLASS_PATH}/tools/util/include
    ./csrc
)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# 链接库路径
# link_directories(
#     /usr/local/cuda/lib64
# )

# 设置编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 --expt-relaxed-constexpr")


option(ENABLE_UNIT_TESTS "Enable unit tests" OFF)


if (ENABLE_UNIT_TESTS)
    add_subdirectory(tests/unit)
endif ()


# add_library(
#     onebit_sparse
#     ./csrc/kernel.cu
# )
add_executable(
    onebit_sparse
    ./csrc/kernel.cu
)

link_libraries(
    onebit_sparse
    cuda cudadevrt cudart_static cublas 
)
set_target_properties(onebit_sparse PROPERTIES
    COMPILE_FLAGS "-g"
    # LINK_FLAGS "2> error.txt"
)

# 定义可执行文件及其源文件
# add_executable(wmma wmma.cu)
# add_executable(swmma sparse_wmma.cu)
# add_executable(tmma tensor_mma_op.cu)
# add_executable(unit unit_test_sparse.cu)

# 链接库
# target_link_libraries(wmma ${CUDA_LIBRARIES} cuda cudadevrt cudart_static cublas)
# target_link_libraries(swmma ${CUDA_LIBRARIES} cuda cudadevrt cudart_static cublas gtest)
# target_link_libraries(tmma ${CUDA_LIBRARIES} cuda cudadevrt cudart_static cublas)
# target_link_libraries(unit ${CUDA_LIBRARIES} cuda cudadevrt cudart_static cublas gtest)

# 将错误输出重定向到文件

# set_target_properties(unit PROPERTIES
#     LINK_FLAGS "2> error.txt"
# )