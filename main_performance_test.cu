#include <iostream>
#include <iomanip>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "secp256k1.cuh"
#include <ctime>

__global__ void generateRandomPrivateKeyKernel(u64* d_privateKeys, unsigned long long seed, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys * 4) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        d_privateKeys[idx] = curand(&state);
    }
}

__global__ void computePublicKey(Point* d_publicKeys, const u64* d_privateKeys, int numKeys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numKeys) {
        secp256k1PublicKey(&d_publicKeys[idx], &d_privateKeys[idx * 4]);
    }
}

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Query the GPU for its properties
    cudaDeviceProp prop;
    checkCudaError(cudaGetDeviceProperties(&prop, 0), "Getting device properties");

    std::cout << "GPU Model: " << prop.name << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;

    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxKeys = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / 256) * 256;  // Estimate based on the GPU's capabilities
    std::cout << "Generating maximum of " << maxKeys << " private keys and corresponding public keys." << std::endl;

    int threadsPerBlock = 256;  // Safe value to avoid resource limit issues
    int blocksForKeys = (maxKeys + threadsPerBlock - 1) / threadsPerBlock;
    int blocksForPrivateKey = (maxKeys * 4 + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory on the device for private keys and public keys
    u64* d_privateKeys;
    checkCudaError(cudaMalloc((void**)&d_privateKeys, maxKeys * 4 * sizeof(u64)), "Allocating device memory for private keys");

    Point* d_publicKeys;
    checkCudaError(cudaMalloc((void**)&d_publicKeys, maxKeys * sizeof(Point)), "Allocating device memory for public keys");

    // Generate random seed using current time
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Creating start event");
    checkCudaError(cudaEventCreate(&stop), "Creating stop event");

    // Start timing
    checkCudaError(cudaEventRecord(start), "Recording start event");

    // Generate random private keys on the device
    generateRandomPrivateKeyKernel<<<blocksForPrivateKey, threadsPerBlock>>>(d_privateKeys, seed, maxKeys);
    checkCudaError(cudaGetLastError(), "Launching generateRandomPrivateKeyKernel");
    checkCudaError(cudaDeviceSynchronize(), "Synchronizing after generateRandomPrivateKeyKernel");

    // Launch kernel to compute public keys
    computePublicKey<<<blocksForKeys, threadsPerBlock>>>(d_publicKeys, d_privateKeys, maxKeys);
    checkCudaError(cudaGetLastError(), "Launching computePublicKey");
    checkCudaError(cudaDeviceSynchronize(), "Synchronizing after computePublicKey");

    // Stop timing
    checkCudaError(cudaEventRecord(stop), "Recording stop event");
    checkCudaError(cudaEventSynchronize(stop), "Synchronizing stop event");

    // Calculate elapsed time
    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time");

    // Calculate keys per second
    float seconds = milliseconds / 1000.0f;
    float keysPerSecond = maxKeys / seconds;

    std::cout << "Generated " << maxKeys << " private keys and public keys in " << seconds << " seconds." << std::endl;
    std::cout << "Keys per second: " << keysPerSecond << std::endl;

    // Free device memory
    checkCudaError(cudaFree(d_privateKeys), "Freeing device memory for private keys");
    checkCudaError(cudaFree(d_publicKeys), "Freeing device memory for public keys");

    // Destroy CUDA events
    checkCudaError(cudaEventDestroy(start), "Destroying start event");
    checkCudaError(cudaEventDestroy(stop), "Destroying stop event");

    return 0;
}
