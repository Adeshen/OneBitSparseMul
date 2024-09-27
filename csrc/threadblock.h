#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"

#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/default_mma_sparse_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"

#include "cutlass/gemm/threadblock/default_mma_core.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "cutlass/gemm/threadblock/mma_sparse_multistage.h"
#include "cutlass/gemm/threadblock/default_mma_core_sparse_sm80.h"
#include "cutlass/gemm/threadblock/default_sparse_mma.h"
#include "warp.h"
#include "onebit_sparse_multistage.h"

namespace cutlass
{

    namespace gemm
    {
        namespace threadblock
        {
            /// Partial specialization:
            ///
            ///   A: column-major
            ///   B: row-major
            ///   Operator: tensor op class
            ///
            /// This uses the default warp-level operator given tile sizes
            template <
                /// Shape of threadblock-scoped matrix multiply operator (concept:
                /// GemmShape)
                typename Shape_,
                /// Shape of warp-level matrix multiply operator (concept: GemmShape)
                typename WarpShape_,
                /// Shape of one matrix production operation (concept: GemmShape)
                /// Data type of A operand
                /// Data type of B operand
                typename ElementB_,
                /// Data type of accumulator
                /// Layout of accumulator
                typename LayoutC_,
                /// Number of stages
                int Stages,
                /// Operation performed by MMA
                /// Cache operation of operand A
                cutlass::arch::CacheOperation::Kind CacheOpA,
                /// Cache operation of operand B
                cutlass::arch::CacheOperation::Kind CacheOpB>
            struct DefaultSparseMmaCore<Shape_,
                                        WarpShape_,
                                        cutlass::gemm::GemmShape<16, 8, 32>,
                                        cutlass::uint1b_t, layout::ColumnMajor,
                                        ElementB_, layout::RowMajor,
                                        float, LayoutC_,
                                        arch::OpClassTensorOp, Stages,
                                        cutlass::arch::OpMultiplyAdd, false, CacheOpA, CacheOpB>
            {
                using Shape = Shape_;
                using WarpShape = WarpShape_;
                using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
                using ElementA = cutlass::uint1b_t;
                using LayoutA = layout::ColumnMajor;
                using ElementB = ElementB_;
                using LayoutB = layout::RowMajor;
                using ElementC = float;
                using LayoutC = LayoutC_;
                static int const kStages = Stages;
                static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
                static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

                static int const kSparse = 2;

                /// Number of warps present
                using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                                            Shape::kN / WarpShape::kN,
                                            Shape::kK / WarpShape::kK>;

                // Divisility requirements
                static_assert(
                    !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

                /// Number of threads per warp
                static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

                /// Number of threads total
                static int const kThreads = WarpCount::kCount * kWarpSize;

                /// Size of a threadblock-scoped access
                static int const kAccessSizeInBits = 128;

                /// Default Operator
                using Operator = cutlass::arch::OpMultiplyAdd;

                // Warp thread arrangement
                static int const kWarpThreadArrangementContiguousA =
                    platform::min(Shape::kM / (32 / sizeof_bits<ElementA>::value), 8);

                static int const kWarpThreadArrangementStridedA =
                    kWarpSize / kWarpThreadArrangementContiguousA;

                static int const kWarpThreadArrangementContiguousB =
                    platform::min(Shape::kN / (kAccessSizeInBits / sizeof_bits<ElementB>::value), 8);

                static int const kWarpThreadArrangementStridedB =
                    kWarpSize / kWarpThreadArrangementContiguousB;

                //
                // Shared memory layouts
                //
                // static int const Crosswise_A = platform::min(int(64 / sizeof(ElementA)),
                //                                              Shape::kM);

                // using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous<
                //     sizeof_bits<ElementA>::value, Crosswise_A>;
                using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous64b;

                // Shared memory layout
                static int const Crosswise_B = platform::min(int(128 / sizeof(ElementB)),
                                                             Shape::kN);

                using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous<
                    sizeof_bits<ElementB>::value, Crosswise_B>;

                //
                // Iterators to write to shared memory
                //

                /// ThreadMap of iterator A
                static_assert(kWarpThreadArrangementContiguousA != 0, "kWarpThreadArrangementContiguousA is zeros");
                using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
                    layout::PitchLinearShape<Shape::kM, Shape::kK / kSparse>, kThreads,
                    layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                             kWarpThreadArrangementStridedA>,
                    32 / sizeof_bits<ElementA>::value>;

                /// Shared memory iterator to A operand
                using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
                    MatrixShape<Shape::kM, Shape::kK / kSparse>, ElementA, SmemLayoutA, 1,
                    IteratorThreadMapA>;

                /// ThreadMap of iterator B
                using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
                    layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
                    layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                             kWarpThreadArrangementStridedB>,
                    kAccessSizeInBits / sizeof_bits<ElementB>::value>;

                /// Shared memory iterator to B operand
                using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
                    MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
                    IteratorThreadMapB>;

                //
                // Warp-level matrix multiply operator
                //

                // Define the warp-level tensor op

                using ArchSparseMma = cutlass::arch::SparseMma<
                    cutlass::gemm::GemmShape<16, 8, 32>,
                    32,
                    cutlass::half_t,
                    cutlass::layout::RowMajor,
                    cutlass::half_t,
                    cutlass::layout::ColumnMajor,
                    float,
                    cutlass::layout::RowMajor,
                    cutlass::arch::OpMultiplyAdd,
                    cutlass::arch::SPFormatType::Thread>;
                using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
                    ArchSparseMma, cutlass::MatrixShape<1, 1>>;

                using MmaTensorOp = typename cutlass::gemm::warp::OneBitSparseMmaTensorOp<
                    WarpShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
                    ElementC, LayoutC, Policy, WarpCount::kK>;

                /// Cache operation of operand E
                static cutlass::arch::CacheOperation::Kind const kCacheOpE =
                    cutlass::arch::CacheOperation::Global;

                static int const kInterleavedE = MmaTensorOp::kInterleaved;
                static int const kMetaSizeInBits = MmaTensorOp::kMetaSizeInBits;
                static int const kMaxID2 = MmaTensorOp::kMaxID2;
                static int const kElementsPerElementE = MmaTensorOp::kElementsPerElementE;

                using ElementE = typename MmaTensorOp::ElementE;
                using GmemLayoutE = cutlass::layout::ColumnMajorInterleaved<kInterleavedE>;

                // Shared memory layout.  Interleaved layout is mapped to PitchLinear layout.
                using SmemLayoutE = typename MmaTensorOp::LayoutE;

                /// ThreadMap of iterator E
                static int const kElementsPerAccessE =
                    kAccessSizeInBits / sizeof_bits<ElementE>::value;

                /// E is tiny.  Not all warps are needed.
                static int const kThreadsE =
                    (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
                         (kAccessSizeInBits / sizeof_bits<ElementE>::value) >
                     kThreads)
                        ? kThreads
                        : (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
                           (kAccessSizeInBits / sizeof_bits<ElementE>::value));

                using IteratorThreadMapE = transform::PitchLinearStripminedThreadMap<
                    layout::PitchLinearShape<Shape::kM * kInterleavedE,
                                             Shape::kK / kSparse / kElementsPerElementE /
                                                 kInterleavedE>,
                    kThreadsE, kElementsPerAccessE>;

                /// Shared memory iterator to E operand
                using SmemIteratorE = transform::threadblock::RegularTileAccessIterator<
                    MatrixShape<Shape::kM * kInterleavedE,
                                Shape::kK / kSparse / kElementsPerElementE / kInterleavedE>,
                    ElementE, SmemLayoutE, 0, IteratorThreadMapE>;

                /// Policy used to define MmaPipelined
                using MmaPolicy =
                    SparseMmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>,
                                    MatrixShape<0, 0>, WarpCount::kK>;
            };

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
                /// Access granularity of B matrix in units of elements
                int kAlignmentB,
                /// Element type for internal accumulation
                typename ElementAccumulator,
                /// Tag indicating architecture to tune for
                typename ArchTag,
                /// Threadblock-level tile size (concept: GemmShape)
                typename ThreadblockShape,
                /// Warp-level tile size (concept: GemmShape)
                typename WarpShape,
                /// Instruction-level tile size (concept: GemmShape)
                typename InstructionShape,
                /// Number of stages used in the multistage mainloop
                int Stages,
                /// Operation perfomed by GEMM
                typename Operator>
            struct DefaultSparseMma<cutlass::uint1b_t, LayoutA, kAlignmentA, ElementB, LayoutB,
                                    kAlignmentB, ElementAccumulator, layout::RowMajor,
                                    arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                                    InstructionShape, Stages, Operator, false>
            {
                // static_assert(false, "ee");
                using ElementA = cutlass::uint1b_t;
                static cutlass::arch::CacheOperation::Kind const CacheOpA =
                    ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
                        ? cutlass::arch::CacheOperation::Global
                        : cutlass::arch::CacheOperation::Always;

                static cutlass::arch::CacheOperation::Kind const CacheOpB =
                    ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
                        ? cutlass::arch::CacheOperation::Global
                        : cutlass::arch::CacheOperation::Always;

                // Define the MmaCore components
                using MmaCore = typename cutlass::gemm::threadblock::DefaultSparseMmaCore<
                    ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
                    ElementB, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
                    Stages, Operator, false, CacheOpA, CacheOpB>;

                static int const kSparse = MmaCore::kSparse;

                // Define iterators over tiles from the A operand
                using ThreadMapA = typename MmaCore::IteratorThreadMapA;
                using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
                using IteratorA =
                    cutlass::transform::threadblock::PredicatedTileAccessIterator<
                        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK / kSparse>,
                        ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

                // Define iterators over tiles from the B operand
                using ThreadMapB = typename MmaCore::IteratorThreadMapB;
                using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
                using IteratorB =
                    cutlass::transform::threadblock::PredicatedTileAccessIterator<
                        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
                        ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

                // Define iterators over tiles from the E operand
                using ElementE = typename MmaCore::ElementE;
                using LayoutE = typename MmaCore::GmemLayoutE;
                using ThreadMapE = typename MmaCore::IteratorThreadMapE;
                using AccessTypeE =
                    cutlass::Array<ElementE, 128 / sizeof_bits<ElementE>::value>;
                using IteratorE =
                    cutlass::transform::threadblock::PredicatedTileAccessIterator<
                        cutlass::MatrixShape<ThreadblockShape::kM,
                                             ThreadblockShape::kK / kSparse /
                                                 MmaCore::kElementsPerElementE>,
                        ElementE, LayoutE, 1, ThreadMapE, AccessTypeE>;

                // Define the threadblock-scoped multistage matrix multiply
                using ThreadblockMma = cutlass::gemm::threadblock::OnebitSparseMmaMultistage<
                    typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
                    MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
                    MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor,
                    IteratorE, typename MmaCore::SmemIteratorE, MmaCore::kCacheOpE,
                    typename MmaCore::MmaPolicy, Stages>;

            }; // namespace threadblock

        }; // namespace gemm
    }
}
