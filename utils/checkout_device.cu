#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1 << 20) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Shared memory per multiprocessor: " << deviceProp.sharedMemPerMultiprocessor << " bytes" << std::endl;
        std::cout << "  Max dynamic shared memory per block: " << deviceProp.sharedMemPerBlockOptin << " bytes" << std::endl;
        std::cout << "  Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << std::endl;
    }

    return 0;
}