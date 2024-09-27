#include <iostream>
#include <vector>
#include "cutlass/numeric_types.h"
#include "cutlass/cutlass.h"
#include "cutlass/arch/mma_sm80.h"
#include <cutlass/layout/layout.h>
#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/gemm/warp/mma_sparse_tensor_op.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_access_iterator.h"
namespace cutlass
{
    namespace gemm
    {
        namespace warp
        {

            template <typename T, int N, FloatRoundStyle Round>
            struct OneBitConvertAndPack
            {
                using Converter = NumericArrayConverter<T, cutlass::uint1b_t, N, Round>;
                using EConverter = NumericConverter<T, cutlass::uint1b_t, Round>;

                CUTLASS_HOST_DEVICE
                Array<T, N> operator()(Array<cutlass::uint1b_t, N> const &source)
                {
                    // Converter converter;
                    // Array<T, N> out_arr = converter(source);
                    // CUTLASS_PRAGMA_UNROLL
                    // for (int i = 0; i < N; ++i)
                    // {
                    //     if (out_arr[i] == T(0))
                    //     {
                    //         out_arr[i] = T(-1);
                    //     }
                    // }
                    // EConverter converter;
                    Array<T, N> out_arr;
                    
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < N; ++i)
                    {   
                        // out_arr[i] = T(1);
                        if (source[i].get())
                        {
                            out_arr[i] = T(1);
                        }else{
                            out_arr[i] = T(-1);
                        }
                        
                    }
                    return out_arr;
                }
            };

            /// Tile access iterator
            /// Each iteration acess in the tile is
            /// used as multiplicand for one
            /// warp-level matrix multiplication
            template <
                /// Size of the tile (concept: MatrixShape)
                typename Shape_,
                /// Operand identity
                Operand Operand_,
                /// Data type of A elements
                typename Element_,
                /// Layout of operand
                typename Layout_,
                /// Shape of one matrix production operation (concept: MatrixShape)
                typename InstructionShape_,
                /// Delta between *MMA operations (in units of *MMA operations, concept:
                /// MatrixShape)
                int OpDelta_,
                /// Number of threads participating in one matrix operation
                int Threads = 32,
                /// Enable Residual Support
                bool EnableResidual = false,
                /// Number of partitions along K dimension
                int PartitionsK_ = 1>
            class OneBitMmaTensorOpMultiplicandTileAccessIterator
            {
            public:
                /// Shape of tile to load (concept: MatrixShape)
                using Shape = Shape_;

                /// Operand tag
                static Operand const kOperand = Operand_;

                /// Basic check
                static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                              "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

                /// Element type
                using Element = Element_;

                /// Layout of source tile
                using Layout = Layout_;

                /// Shape of one matrix product operation (concept: MatrixShape)
                using InstructionShape = InstructionShape_;

                /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
                static int const kOpDelta = OpDelta_;

                /// Number of participating threads
                static int const kThreads = 32;

                /// TensorRef type for loading element from a tensor
                using TensorRef = TensorRef<Element, Layout>;

                /// Index type
                using Index = typename TensorRef::Index;

                /// Long Index type
                using LongIndex = typename TensorRef::LongIndex;

                /// Coordinate for an element in the tensor
                using TensorCoord = typename TensorRef::TensorCoord;

                /// Number of elements accessed per Shared Memory load
                static int const kElementsPerAccess =
                    (sizeof_bits<Element>::value >= 32 ? 1 : 32 / sizeof_bits<Element>::value);

                using InstructionCount = MatrixShape<
                    Shape::kRow / InstructionShape::kRow,
                    Shape::kColumn / InstructionShape::kColumn>;

                static int const kIterations = (kOperand == Operand::kA) ? InstructionCount::kColumn : InstructionCount::kRow;

            public:
                //
                // Derived quantities
                //

                /// Fragment object holding a thread's part of a tile
                using Fragment = Array<
                    Element,
                    (kOperand == Operand::kA) ? (Shape::kRow *InstructionShape::kColumn / kThreads) : (Shape::kColumn *InstructionShape::kRow / kThreads)>;

                /// Memory access type
                using AccessType = AlignedArray<Element, kElementsPerAccess>;

            private:
                /// Underlying tensor reference
                TensorRef ref_;

                /// Extent of tensor
                MatrixCoord extent_;

                /// Origin
                MatrixCoord origin_;

                /// Used to load residual tile
                bool is_residual_;

                /// residual offset of each thread
                TensorCoord residual_offset_;

                /// Iterations in a tile
                int iterations_;

            public:
                /// Constructor from TensorRef
                CUTLASS_HOST_DEVICE
                OneBitMmaTensorOpMultiplicandTileAccessIterator(
                    TensorRef const &ref,
                    TensorCoord extent,
                    int lane_id) : ref_(ref), extent_(extent), is_residual_(false), iterations_(0)
                {

                    if (kOperand == Operand::kA)
                    {
                        origin_ = MatrixCoord(lane_id / 4, (lane_id % 4) * kElementsPerAccess);
                    }
                    else
                    {
                        origin_ = MatrixCoord((lane_id % 4) * kElementsPerAccess, lane_id / 4);
                    }

                    ref_.add_coord_offset(origin_);

                    if (EnableResidual)
                    {
                        // compute residual offset
                        if (kOperand == Operand::kA)
                        {
                            typename TensorCoord::Index residual_size =
                                extent_.column() % Shape::kColumn;
                            if (residual_size)
                            {
                                is_residual_ = true;
                                residual_offset_ = make_Coord(0, residual_size);
                            }
                        }
                        else
                        {
                            typename TensorCoord::Index residual_size =
                                extent_.row() % Shape::kRow;
                            if (residual_size)
                            {
                                is_residual_ = true;
                                residual_offset_ = make_Coord(residual_size, 0);
                            }
                        }
                    }
                }

                /// Constructor from TensorRef
                CUTLASS_HOST_DEVICE
                OneBitMmaTensorOpMultiplicandTileAccessIterator(
                    TensorRef const &ref,
                    int lane_id) : OneBitMmaTensorOpMultiplicandTileAccessIterator(ref,
                                                                                   {Shape::kRow, Shape::kColumn}, lane_id)
                {
                }

                /// Advances an iterator along logical dimensions of matrix in units of whole tiles
                CUTLASS_HOST_DEVICE
                OneBitMmaTensorOpMultiplicandTileAccessIterator &add_tile_offset(TensorCoord const &tile_offset)
                {

                    TensorCoord coord_offset(tile_offset.row() * Shape::kRow, tile_offset.column() * Shape::kColumn);
                    origin_ += coord_offset;

                    ref_.add_coord_offset(coord_offset);

                    return *this;
                }

                /// Advances the iterator along the advance dimension
                CUTLASS_DEVICE
                void advance()
                {

                    if (EnableResidual && is_residual_)
                    {
                        is_residual_ = false;

                        origin_ += residual_offset_;
                        ref_.add_coord_offset(residual_offset_);
                    }

                    else
                    {
                        if (kOperand == Operand::kA)
                        {
                            add_tile_offset({0, 1});
                        }
                        else
                        {
                            add_tile_offset({1, 0});
                        }
                    }

                    iterations_ = 0;
                }

                /// increase iterations in a tile
                CUTLASS_HOST_DEVICE
                OneBitMmaTensorOpMultiplicandTileAccessIterator &operator++()
                {

                    iterations_++;

                    if (iterations_ >= kIterations)
                        advance();

                    return *this;
                }

                /// Loads a fragment from memory at the location pointed to by the iterator.
                CUTLASS_HOST_DEVICE
                void load(Fragment &frag) const
                {

                    int const kWarpShapeDivisibleInner =
                        (kOperand == Operand::kA ? InstructionShape::kColumn : InstructionShape::kRow);

                    // Take advantage of Tensor Op's 8 x 4T access pattern
                    int const kAccessesInner = (kWarpShapeDivisibleInner / kElementsPerAccess) / 4;

                    AccessType *access_ptr = reinterpret_cast<AccessType *>(&frag);

                    if (kOperand == Operand::kA)
                    {
                        int const kTilesPerInstruction = InstructionShape::kRow / 8;

                        CUTLASS_PRAGMA_UNROLL
                        for (int inst_m_idx = 0; inst_m_idx < InstructionCount::kRow; ++inst_m_idx)
                        {

                            CUTLASS_PRAGMA_UNROLL
                            for (int inner_idx = 0; inner_idx < kAccessesInner; ++inner_idx)
                            {

                                CUTLASS_PRAGMA_UNROLL
                                for (int access_m_idx = 0; access_m_idx < kTilesPerInstruction; ++access_m_idx)
                                {
                                    int access_idx =
                                        access_m_idx + kTilesPerInstruction * (inner_idx + kAccessesInner * inst_m_idx);

                                    MatrixCoord offset(
                                        access_m_idx * 8 + inst_m_idx * InstructionShape::kRow,
                                        inner_idx * 4 * kElementsPerAccess + iterations_ * InstructionShape::kColumn);

                                    MatrixCoord access_coord = origin_ + offset;

                                    //            if(access_coord.row() < extent_.row() && access_coord.column() < extent_.column()) {

                                    access_ptr[access_idx] = *reinterpret_cast<AccessType const *>(
                                        ref_.data() + ref_.offset(offset));
                                    //            }
                                    //            else {
                                    //              AccessType zero;
                                    //              zero.clear();
                                    //              access_ptr[access_idx] = zero;
                                    //            }
                                }
                            }
                        }
                    }
                    else
                    {
                        CUTLASS_PRAGMA_UNROLL
                        for (int inst_n_idx = 0; inst_n_idx < InstructionCount::kColumn; ++inst_n_idx)
                        {

                            CUTLASS_PRAGMA_UNROLL
                            for (int inner_idx = 0; inner_idx < kAccessesInner; ++inner_idx)
                            {
                                int access_idx = inner_idx + kAccessesInner * inst_n_idx;

                                MatrixCoord offset(
                                    inner_idx * 4 * kElementsPerAccess + iterations_ * InstructionShape::kRow,
                                    inst_n_idx * 8);

                                MatrixCoord access_coord = origin_ + offset;

                                //          if(access_coord.row() < extent_.row() && access_coord.column() < extent_.column()) {

                                access_ptr[access_idx] = *reinterpret_cast<AccessType const *>(
                                    ref_.data() + ref_.offset(offset));
                                //          }
                                //          else {
                                //              AccessType zero;
                                //              zero.clear();
                                //              access_ptr[access_idx] = zero;
                                //          }
                            }
                        }
                    }
                }
                CUTLASS_DEVICE
                void set_kgroup_index(int k_group)
                {
                    // no operation
                }
            };

            template <
                /// Size of the Gemm problem - concept: gemm::GemmShape<>
                typename Shape_,
                /// Data type of A elements
                typename ElementA_,
                /// Layout of A matrix (concept: MatrixLayout)
                typename LayoutA_,
                /// Data type of B elements
                typename ElementB_,
                /// Layout of B matrix (concept: MatrixLayout)
                typename LayoutB_,
                /// Element type of C matrix
                typename ElementC_,
                /// Layout of C matrix (concept: MatrixLayout)
                typename LayoutC_,
                /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
                typename Policy_,
                /// Number of partitions along K dimension
                int PartitionsK_ = 1,
                /// Store the accumulators in row major or column major.  Row major is used
                /// when output layout is interleaved.
                bool AccumulatorsInRowMajor = false,
                /// Used for partial specialization
                typename Enable = bool>
            class OneBitSparseMmaTensorOp
            {
            public:
                /// Shape of warp-level matrix operation (concept: GemmShape)
                using Shape = Shape_;

                /// Data type of multiplicand A
                using ElementA = ElementA_;

                /// Layout of multiplicand A
                using LayoutA = LayoutA_;

                /// Data type of multiplicand B
                using ElementB = ElementB_;

                /// Layout of multiplicand B
                using LayoutB = LayoutB_;

                /// Data type of accumulator matrix C
                using ElementC = ElementC_;

                /// Layout of accumulator matrix C
                using LayoutC = LayoutC_;

                /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
                using Policy = Policy_;

                /// Equivalant base dense mma
                // using Base = MmaTensorOp<Shape, ElementA, LayoutA, ElementB, LayoutB,
                //                          ElementC, LayoutC, Policy, PartitionsK_,
                //                          AccumulatorsInRowMajor, Enable>;

                /// Underlying matrix multiply operator (concept: arch::Mma)
                using ArchMmaOperator = typename Policy::Operator;

                /// Indicates math operator
                using MathOperator = typename ArchMmaOperator::Operator;

                /// Architecture tag from underlying instruction
                using ArchTag = typename ArchMmaOperator::ArchTag;

                /// Indicates class of matrix operator
                using OperatorClass = arch::OpClassTensorOp;

                /// Shape of underlying instruction
                using InstructionShape = typename ArchMmaOperator::Shape;

                /// Complex transform on A operand
                static ComplexTransform const kTransformA = ComplexTransform::kNone;

                /// Complex transform on B operand
                static ComplexTransform const kTransformB = ComplexTransform::kNone;

                /// Number of threads participating in warp-level matrix product
                static int const kThreadCount = 32;

                /// Number of partitions along K dimension
                static int const kPartitionsK = PartitionsK_;

                /// Sparsity in Operand A
                static int const kSparse = Policy::Operator::kSparse;

                /// Meta data size in bits
                static int const kMetaSizeInBits = Policy::Operator::kMetaSizeInBits;

                /// Max ID2
                static int const kMaxID2 = Policy::Operator::kMaxID2;

                static int const kVerticalVisit = false;
                /// Data type of meta E that is moved at the same time
                using ElementE =
                    typename cutlass::platform::conditional<kMaxID2 == 1, uint32_t,
                                                            uint16_t>::type;

                /// Number of ElementA that is associated with one ElementE
                static int const kElementsPerElementE =
                    128 / cutlass::sizeof_bits<half_t>::value;

                /// Meta data is essentially interleaved but mapped to ColumnMajor internally
                static int const kInterleaved = 2;

                /// Layout of meta E
                using LayoutE = cutlass::layout::ColumnMajor;

            public:
                /// Iterates over the A operand in memory
                using IteratorA = OneBitMmaTensorOpMultiplicandTileAccessIterator<
                    MatrixShape<Shape::kM, Shape::kK / kSparse>, Operand::kA, ElementA,
                    LayoutA,
                    MatrixShape<Policy::Operator::Shape::kM,
                                Policy::Operator::Shape::kK / kSparse>,
                    Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;
                // using IteratorA = MmaTensorOpMultiplicandTileIterator<
                //     MatrixShape<Shape::kM, Shape::kK / kSparse>, Operand::kA, ElementA,
                //     LayoutA,
                //     MatrixShape<Policy::Operator::Shape::kM,
                //                 Policy::Operator::Shape::kK / kSparse>,
                //     Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

                /// Storage for A tile
                using FragmentA = typename IteratorA::Fragment;

                /// Storage for transformed A tile
                using TransformedFragmentA =
                    Array<typename Policy::Operator::ElementA, FragmentA::kElements>;

                /// Iterates over the B operand in memory
                using IteratorB = MmaTensorOpMultiplicandTileIterator<
                    MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
                    MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>,
                    Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

                /// Storage for B tile
                using FragmentB = typename IteratorB::Fragment;

                /// Storage for transformed B tile
                using TransformedFragmentB = Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;

                /// Iterates over the C operand in memory
                using IteratorC = MmaTensorOpAccumulatorTileIterator<
                    MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
                    typename ArchMmaOperator::Shape, typename Policy::OpDelta>;

                /// Storage for C tile
                using FragmentC = typename IteratorC::Fragment;

                /// Iterates over the E operand in memory
                using IteratorE = SparseMmaTensorOpMetaTileIterator<
                    MatrixShape<Shape::kM * kInterleaved,
                                Shape::kK / kSparse / kElementsPerElementE / kInterleaved>,
                    ElementE, LayoutE,
                    MatrixShape<Policy::Operator::Shape::kM,
                                Policy::Operator::Shape::kK / kSparse / kElementsPerElementE /
                                    kInterleaved>,
                    Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

                /// Storage for E tile
                using FragmentE = typename IteratorE::Fragment;

                /// Number of mma operations performed
                using MmaIterations = MatrixShape<
                    (Shape::kM + ArchMmaOperator::Shape::kM - 1) / ArchMmaOperator::Shape::kM,
                    (Shape::kN + ArchMmaOperator::Shape::kN - 1) / ArchMmaOperator::Shape::kN>;

            public:
                /// Underlying matrix multiply operator (concept: arch::Mma)
                ArchMmaOperator mma;

            public:
                //
                // Methods
                //

                /// Ctor
                CUTLASS_DEVICE
                OneBitSparseMmaTensorOp() {}

                /// Performs a warp-level matrix multiply-accumulate operation
                CUTLASS_DEVICE
                void operator()(
                    FragmentC &D,
                    TransformedFragmentA const &A,
                    TransformedFragmentB const &B,
                    FragmentC const &C,
                    FragmentE const &E) const
                {

                    using MmaOperandA = typename Policy::Operator::FragmentA;
                    using MmaOperandB = typename Policy::Operator::FragmentB;
                    using MmaOperandC = typename Policy::Operator::FragmentC;
                    using MmaOperandE = typename Policy::Operator::FragmentE;

                    D = C;

                    MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
                    MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
                    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);
                    MmaOperandE const *ptr_E = reinterpret_cast<MmaOperandE const *>(&E);

                    if (kVerticalVisit)
                    {
                        CUTLASS_PRAGMA_UNROLL
                        for (int n = 0; n < MmaIterations::kColumn; ++n)
                        {

                            CUTLASS_PRAGMA_UNROLL
                            for (int m = 0; m < MmaIterations::kRow; ++m)
                            {

                                int m_serpentine = ((n % 2) ? (MmaIterations::kRow - 1 - m) : m);
                                int id2 = m_serpentine % kMaxID2;
                                if (AccumulatorsInRowMajor)
                                { // matrix B is reordered
                                    mma(
                                        ptr_D[n + m_serpentine * MmaIterations::kColumn],
                                        ptr_A[m_serpentine],
                                        ptr_B[n],
                                        ptr_D[n + m_serpentine * MmaIterations::kColumn],
                                        ptr_E[(m_serpentine / kMaxID2)],
                                        id2);
                                }
                                else
                                {
                                    mma(
                                        ptr_D[m_serpentine + n * MmaIterations::kRow],
                                        ptr_A[m_serpentine],
                                        ptr_B[n],
                                        ptr_D[m_serpentine + n * MmaIterations::kRow],
                                        ptr_E[(m_serpentine / kMaxID2)],
                                        id2);
                                }
                            }
                        }
                    }
                    else
                    {
                        CUTLASS_PRAGMA_UNROLL
                        for (int m = 0; m < MmaIterations::kRow; ++m)
                        {

                            int id2 = m % kMaxID2;

                            CUTLASS_PRAGMA_UNROLL
                            for (int n = 0; n < MmaIterations::kColumn; ++n)
                            {

                                int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);

                                if (AccumulatorsInRowMajor)
                                { // matrix B is reordered
                                    mma(
                                        ptr_D[n_serpentine + m * MmaIterations::kColumn],
                                        ptr_A[m],
                                        ptr_B[n_serpentine],
                                        ptr_D[n_serpentine + m * MmaIterations::kColumn],
                                        ptr_E[(m / kMaxID2)],
                                        id2);
                                }
                                else
                                {
                                    mma(ptr_D[m + n_serpentine * MmaIterations::kRow],
                                        ptr_A[m],
                                        ptr_B[n_serpentine],
                                        ptr_D[m + n_serpentine * MmaIterations::kRow],
                                        ptr_E[(m / kMaxID2)],
                                        id2);
                                }
                            }
                        }
                    }
                }

                /// Transform the mma operands to the required types
                CUTLASS_DEVICE
                void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                               FragmentA const &A, FragmentB const &B) const
                {

                    //
                    // Define conversions from source type to instruction type
                    //
                    FloatRoundStyle const kRoundA =
                        PreferredRoundingMode<typename ArchMmaOperator::ElementA,
                                              ElementA>::kRound;
                    FloatRoundStyle const kRoundB =
                        PreferredRoundingMode<typename ArchMmaOperator::ElementB,
                                              ElementB>::kRound;

                    if (kVerticalVisit)
                    {
                        detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                                               FragmentA::kElements, kRoundA>
                            convert_A;
                        NumericArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                                              FragmentB::kElements / 2, kRoundB>
                            convert_B;
                        Array<ElementB, FragmentB::kElements / 2> const *ptr_B =
                            reinterpret_cast<Array<ElementB, FragmentB::kElements / 2> const *>(&B);
                        Array<typename ArchMmaOperator::ElementB, FragmentB::kElements / 2> *
                            ptr_dst_B = reinterpret_cast<Array<typename ArchMmaOperator::ElementB,
                                                               FragmentB::kElements / 2> *>(&dst_B);

                        dst_A = convert_A(A);

                        ptr_dst_B[0] = convert_B(ptr_B[0]);
                        ptr_dst_B[1] = convert_B(ptr_B[1]);
                    }
                    else
                    {
                        OneBitConvertAndPack<typename ArchMmaOperator::ElementA,
                                             FragmentA::kElements / 2, kRoundA>
                            convert_A;
                        // detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                        //                      FragmentA::kElements / 2, kRoundA>
                        //     convert_A;
                            // detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                            //                  FragmentA::kElements / 2, kRoundA>::Converter b;
                            //                  b++;
                        NumericArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                                              FragmentB::kElements, kRoundB>
                            convert_B;
                        Array<ElementA, FragmentA::kElements / 2> const *ptr_A =
                            reinterpret_cast<Array<ElementA, FragmentA::kElements / 2> const *>(&A);
                        Array<typename ArchMmaOperator::ElementA, FragmentA::kElements / 2> *
                            ptr_dst_A = reinterpret_cast<Array<typename ArchMmaOperator::ElementA,
                                                               FragmentA::kElements / 2> *>(&dst_A);

                        dst_B = convert_B(B);

                        ptr_dst_A[0] = convert_A(ptr_A[0]);
                        ptr_dst_A[1] = convert_A(ptr_A[1]);
                    }
                }
            };

        }
    }

}

namespace onebit
{

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

    using ElementA = cutlass::uint1b_t;
    // using ElementA = int8_t;
    using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<ElementA>::value, 128>;
    using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<cutlass::half_t>::value, 64>;

    using WarpSparseMma = cutlass::gemm::warp::OneBitSparseMmaTensorOp<
        cutlass::gemm::GemmShape<32, 32, 64>,
        ElementA,
        LayoutA,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        Policy>;

};
