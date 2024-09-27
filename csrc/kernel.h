#pragma once

#include "cutlass/cutlass.h"
#include "threadblock.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/kernel/default_gemm_sparse.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_sparse.h"
// #include "onebit_sparse_multistage.h"
namespace cutlass
{
    namespace gemm
    {
        namespace kernel
        {
            /// Partial specialization for Ampere Architecture
            template <
                /// Element type for A matrix operand
                /// Layout type for A matrix operand
                typename LayoutA,
                /// Access granularity of A matrix in units of elements
                int kAlignmentA,
                /// Element type for B matrix operand
                typename ElementB,
                /// Layout type for B matrix operand
                typename LayoutB,
                /// Access granularity of A matrix in units of elements
                int kAlignmentB,
                /// Element type for C and D matrix operands
                typename ElementC,
                /// Element type for internal accumulation
                typename ElementAccumulator,
                /// Threadblock-level tile size (concept: GemmShape)
                typename ThreadblockShape,
                /// Warp-level tile size (concept: GemmShape)
                typename WarpShape,
                /// Warp-level tile size (concept: GemmShape)
                typename InstructionShape,
                /// Epilogue output operator
                typename EpilogueOutputOp,
                /// Threadblock-level swizzling operator
                typename ThreadblockSwizzle,
                /// Number of stages used in the pipelined mainloop
                int Stages,
                /// If true, kernel is configured to support serial reduction in the
                /// epilogue
                bool SplitKSerial,
                /// Operation performed by GEMM
                typename Operator>
            struct DefaultSparseGemm<cutlass::uint1b_t, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC,
                                     layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
                                     arch::Sm80, ThreadblockShape, WarpShape, InstructionShape,
                                     EpilogueOutputOp, ThreadblockSwizzle, Stages, SplitKSerial,
                                     Operator>
            {
                /// Define the threadblock-scoped matrix multiply-accumulate
                using Mma = typename cutlass::gemm::threadblock::DefaultSparseMma<
                    cutlass::uint1b_t, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
                    ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, arch::Sm80,
                    ThreadblockShape, WarpShape, InstructionShape, Stages,
                    Operator>::ThreadblockMma;

                static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

                /// Define the epilogue
                using Epilogue =
                    typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
                        ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
                        EpilogueOutputOp::kCount>::Epilogue;
                
                
                /// Define the kernel-level GEMM operator.
                using GemmKernel = kernel::SparseGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
            };
        }
    }

} // namespace

namespace onebit
{
    namespace device
    {
        using ElementA = cutlass::uint1b_t;
        using LayoutA = cutlass::layout::ColumnMajor;
        using ElementB = cutlass::half_t;
        using LayoutB = cutlass::layout::RowMajor;
        using ElementC = float;
        using LayoutC = cutlass::layout::RowMajor;

        constexpr int32_t kAlignmentA = 32 / cutlass::sizeof_bits<ElementA>::value;
        constexpr int32_t kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

        // using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 128>;
        // using WarpShape = cutlass::gemm::GemmShape<128, 32, 128>;

        using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 128>;
        using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
        using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

        using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
            ElementC,
            128 / cutlass::sizeof_bits<ElementC>::value,
            ElementC,
            ElementC>;
        using KernelGemm = cutlass::gemm::kernel::DefaultSparseGemm<
            ElementA,
            LayoutA,
            kAlignmentA,
            ElementB,
            LayoutB,
            kAlignmentB,
            ElementC,
            LayoutC,
            ElementC,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOutputOp,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
            4,
            false,
            cutlass::arch::OpMultiplyAdd>;

        using DeviceSparseGemm = cutlass::gemm::device::SparseGemm<
            ElementA,
            LayoutA,
            ElementB,
            LayoutB,
            ElementC,
            LayoutC,
            ElementC,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOutputOp,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            4,
            kAlignmentA,
            kAlignmentB,
            false,
            cutlass::arch::OpMultiplyAdd>;
    }

} // namespace onebit

#define CUTLASS_CHECK(status)                                                                          \
    {                                                                                                  \
        cutlass::Status error = status;                                                                \
        if (error != cutlass::Status::kSuccess)                                                        \
        {                                                                                              \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                      << std::endl;                                                                    \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    }
