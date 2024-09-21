#pragma once

#include "cutlass/cutlass.h"
#include "threadblock.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/kernel/default_gemm_sparse.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_sparse.h"

namespace onebit
{
    namespace device{
        using ElementA = cutlass::uint1b_t;
        using LayoutA = cutlass::layout::ColumnMajor;
        using ElementB = cutlass::half_t;
        using LayoutB = cutlass::layout::RowMajor;
        using ElementC = float;
        using LayoutC = cutlass::layout::RowMajor;

        constexpr int32_t kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
        constexpr int32_t kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

        using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 256>;
        using WarpShape = cutlass::gemm::GemmShape<128, 32, 128>;
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
                                                       cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
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
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
            4,
            kAlignmentA,
            kAlignmentB,
            false,
            cutlass::arch::OpMultiplyAdd>;
    }

} // namespace onebit

CUTLASS_DEVICE
void onebit_sparse_matmul(void *a, void *b, void *c, void *d);
