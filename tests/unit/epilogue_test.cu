#include <iostream>
#include "kernel.h"
#include "gtest/gtest.h"


#include "cutlass/aligned_buffer.h"
#include "cutlass/half.h"
#include "cutlass/complex.h"
#include "cutlass/quaternion.h"
#include "cutlass/platform/platform.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "unit/epilogue/threadblock/testbed.h"

TEST(SM80_Sparsekernel_epilogue, 1 ){

    using Epilogue = onebit::device::KernelGemm::Epilogue ;

    EpilogueTestbed<Epilogue> testbed;
    bool result=testbed.run_all();
    printf("%d", result);
    EXPECT_TRUE(result);
}