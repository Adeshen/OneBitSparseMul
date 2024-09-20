#include <iostream>
#include <vector>
#include <gtest/gtest.h>
// #include <cuda_runtime.h>
// #include <cuda_bf16.h>
// #include "cutlass/numeric_types.h"
// #include "cutlass/cutlass.h"
// #include "cutlass/arch/mma_sm80.h"
#include <cutlass/layout/layout.h>
// #include "cutlass/gemm/warp/mma_tensor_op_policy.h"
// #include "cutlass/gemm/warp/mma_sparse_tensor_op.h"
// #include "cutlass/util/host_tensor.h"
// #include "cutlass/arch/arch.h"
#include "warp_test.h"


// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
  EXPECT_EQ(7 * 6, 42);
}

TEST(HelloTest, BasicAssertions2) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
  EXPECT_EQ(7 * 6, 42);
}

TEST(SparseWarpTensorOp, SparseWarpTensorOpxxx)
{
    // EXPECT_EQ(7 * 6, 43);
    using ThreadBlockShape = cutlass::gemm::GemmShape<32, 32, 64>;
    onebit::test::SparseTestbed<onebit::WarpSparseMma, ThreadBlockShape, cutlass::arch::OpMultiplyAdd>().run();
}
