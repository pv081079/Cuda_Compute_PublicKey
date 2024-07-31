#include <iostream>
#include <iomanip>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "secp256k1.cuh"
#include <ctime>

__global__ void computePublicKey(Point* d_publicKey, const u64* d_privateKey) {
    secp256k1PublicKey(d_publicKey, d_privateKey);
}

int main() {
    // Allocate memory on the device for private key and public key
    u64* d_privateKey;
    cudaMalloc((void**)&d_privateKey, 4 * sizeof(u64));

    Point* d_publicKey;
    cudaMalloc((void**)&d_publicKey, sizeof(Point));

    // Prompt the user for the private key
    u64 h_privateKey[4];
    std::cout << "Enter the private key (64 hexadecimal characters): ";
    std::string privateKeyStr;
    std::cin >> privateKeyStr;

    // Convert the string input to u64 array
    for (int i = 0; i < 4; ++i) {
        std::stringstream ss;
        ss << std::hex << privateKeyStr.substr(i * 16, 16);
        ss >> h_privateKey[3 - i];  // Little endian order
    }

    // Copy the private key to the device
    cudaMemcpy(d_privateKey, h_privateKey, 4 * sizeof(u64), cudaMemcpyHostToDevice);

    // Launch kernel to compute public key
    computePublicKey<<<1, 1>>>(d_publicKey, d_privateKey);
    cudaDeviceSynchronize();

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
