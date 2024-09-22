#include "cutlass/cutlass.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"



int main(){
    const cutlass::AlignedArray<float, 2, 8> a();
    const cutlass::AlignedArray<float, 2, 8> b();

    a = b;

    
    
}