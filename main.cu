#include <iostream>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>
#include "secp256k1.cuh"

// Kernel to compute public key from private key
__global__ void computePublicKey(Point* d_publicKey, const u64* d_privateKey) {
    secp256k1PublicKey(d_publicKey, d_privateKey);
}

void generateRandomPrivateKey(u64* privateKey) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<u64> dis;

    for (int i = 0; i < 4; ++i) {
        privateKey[i] = dis(gen);
    }
}

int main() {
    // Generate random private key
    u64 h_privateKey[4];
    generateRandomPrivateKey(h_privateKey);

    // Allocate memory on the device
    u64* d_privateKey;
    cudaMalloc((void**)&d_privateKey, 4 * sizeof(u64));
    cudaMemcpy(d_privateKey, h_privateKey, 4 * sizeof(u64), cudaMemcpyHostToDevice);

    Point* d_publicKey;
    cudaMalloc((void**)&d_publicKey, sizeof(Point));

    // Launch kernel
    computePublicKey<<<1, 1>>>(d_publicKey, d_privateKey);

    // Copy the result back to the host
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
