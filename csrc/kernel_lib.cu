#include "cuda_runtime.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/layout.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/cutlass.h"
#include "kernel.h"
#include "cutlass/util/device_memory.h"

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

CUTLASS_DEVICE
void onebit_sparse_matmul(void *a, void *b, void *c, void *d, void *e, int m, int n, int k)
{

    using Gemm = onebit::device::DeviceSparseGemm;

    Gemm::ElementA *A = static_cast<Gemm::ElementA *>(a);
    Gemm::ElementB *B = static_cast<Gemm::ElementB *>(b);
    Gemm::ElementC *C = static_cast<Gemm::ElementC *>(c);
    Gemm::ElementC *D = static_cast<Gemm::ElementC *>(d);
    Gemm::ElementE *E = static_cast<Gemm::ElementE *>(e);

    constexpr int kSparse = Gemm::kSparse;
    // How many elements of A are covered per ElementE
    constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
    cutlass::TensorRef const ref_a(A, Gemm::LayoutA::packed({m, k / kSparse}));
    cutlass::TensorRef const ref_b(B, Gemm::LayoutB::packed({k, n}));
    cutlass::TensorRef const ref_c(C, Gemm::LayoutC::packed({m, n}));
    cutlass::TensorRef ref_d(D, Gemm::LayoutC::packed({m, n}));
    cutlass::TensorRef const ref_e(E, Gemm::LayoutE::packed({m, k / kSparse / kElementsPerElementE}));

    float alpha = 1;
    float beta = 1;
    int split_k_slices = 1;

    cutlass::gemm::GemmCoord problem_size(m, n, k);
    Gemm::Arguments const arguments{problem_size,    // <- problem size of matrix multiplication
                                    ref_a,           // <- reference to matrix A on device
                                    ref_b,           // <- reference to matrix B on device
                                    ref_c,           // <- reference to matrix C on device
                                    ref_d,           // <- reference to matrix D on device
                                    ref_e,           // <- reference to matrix E on device
                                    {alpha, beta},   // <- tuple of alpha and beta
                                    split_k_slices}; // <- k-dimension split factor
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    // Launch initialized CUTLASS kernel
    status = gemm_op();
    CUTLASS_CHECK(status);
}