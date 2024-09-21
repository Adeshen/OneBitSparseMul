
#include "threadblock.h"
#include "unit/gemm/threadblock/mma_multistage_sparse_testbed.h"
#include "threadblock_test.h"


TEST(SM80_sparse_gemm_threadblock_congruous,
     tensor_op_128x64x256_128x32x128_16x8x32_4stage) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 512);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 256>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 4;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultSparseMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp,
      Stages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

//   test::gemm::threadblock::SparseTestbed<MmaCore>(
//       problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
//       .run(grid, block);
  onebit::test::threadblock::SparseTestbed<MmaCore>(
      problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
      .run(grid, block);
}