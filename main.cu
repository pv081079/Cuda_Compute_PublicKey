#include <iostream>
#include <iomanip>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "secp256k1.cuh"
#include <ctime>

__global__ void generateRandomPrivateKeyKernel(u64* d_privateKey, unsigned long long seed) {
    int idx = threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state);

    d_privateKey[idx] = curand(&state);
}

__global__ void computePublicKey(Point* d_publicKey, const u64* d_privateKey) {
    secp256k1PublicKey(d_publicKey, d_privateKey);
}

int main() {
    // Allocate memory on the device for private key and public key
    u64* d_privateKey;
    cudaMalloc((void**)&d_privateKey, 4 * sizeof(u64));

    Point* d_publicKey;
    cudaMalloc((void**)&d_publicKey, sizeof(Point));

    // Generate random seed using current time
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));

    // Generate random private key on the device
    generateRandomPrivateKeyKernel<<<1, 4>>>(d_privateKey, seed);
    cudaDeviceSynchronize();

    // Launch kernel to compute public key
    computePublicKey<<<1, 1>>>(d_publicKey, d_privateKey);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    u64 h_privateKey[4];
    cudaMemcpy(h_privateKey, d_privateKey, 4 * sizeof(u64), cudaMemcpyDeviceToHost);

    Point h_publicKey;
    cudaMemcpy(&h_publicKey, d_publicKey, sizeof(Point), cudaMemcpyDeviceToHost);

    // Print the private key
    std::cout << "Private Key: ";
    for (int i = 3; i >= 0; --i) {
        std::cout << std::hex << std::setfill('0') << std::setw(16) << h_privateKey[i];
    }
    std::cout << std::endl;

    // Print the public key
    std::cout << "Public Key:" << std::endl;
    std::cout << "X: ";
    for (int i = 3; i >= 0; --i) {
        std::cout << std::hex << std::setfill('0') << std::setw(16) << h_publicKey.x[i];
    }
    std::cout << std::endl;

    std::cout << "Y: ";
    for (int i = 3; i >= 0; --i) {
        std::cout << std::hex << std::setfill('0') << std::setw(16) << h_publicKey.y[i];
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_privateKey);
    cudaFree(d_publicKey);

    return 0;
}
