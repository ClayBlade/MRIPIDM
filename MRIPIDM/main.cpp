#include <iostream>
#include <cuda_runtime.h>
#include "BlochKernelGMGPU.h" // Your kernel file

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    const int SpinMxX = 1;
    const int SpinMxY = 1;
    const int SpinMxZ = 1;
    const int SpinNum = 1;
    const int TypeNum = 1;
    const int TxCoilNum = 1;
    const int RxCoilNum = 1;
    const int SeqLen = 1;
    const int SignalLen = 1;
    const int SBufferLen = 1;

    const int totalSpins = SpinMxX * SpinMxY * SpinMxZ * SpinNum * TypeNum;

    float *d_Mx, *d_My, *d_Mz;
    CHECK_CUDA(cudaMalloc(&d_Mx, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_My, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_Mz, sizeof(float) * totalSpins));

    float *d_Rho, *d_T1, *d_T2, *d_K, *d_dB0, *d_dWRnd;
    CHECK_CUDA(cudaMalloc(&d_Rho, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_T1, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_T2, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_K, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_dB0, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_dWRnd, sizeof(float) * totalSpins));

    // Dummy values
    float Gyro = 42.58e6f;
    float RxCoilDefault = 1.0f, TxCoilDefault = 1.0f;
    float rfRef = 0;

    float *d_CS, *d_TypeFlag, *d_Buffer, *d_Sig, *d_Sx, *d_Sy;
    float *d_Gzgrid, *d_Gygrid, *d_Gxgrid, *d_TxCoilmg, *d_TxCoilpe, *d_RxCoilx, *d_RxCoily;
    float *d_b_Mx, *d_b_My, *d_b_Mz;

    CHECK_CUDA(cudaMalloc(&d_CS, sizeof(double) * TypeNum));
    CHECK_CUDA(cudaMalloc(&d_TypeFlag, sizeof(double) * TypeNum));
    CHECK_CUDA(cudaMalloc(&d_Buffer, sizeof(float) * 6 * TypeNum * SpinMxX * SpinMxY));
    CHECK_CUDA(cudaMalloc(&d_Sig, sizeof(float) * SeqLen * (5 + 3 * TxCoilNum)));
    CHECK_CUDA(cudaMalloc(&d_Sx, sizeof(float) * SignalLen));
    CHECK_CUDA(cudaMalloc(&d_Sy, sizeof(float) * SignalLen));
    CHECK_CUDA(cudaMalloc(&d_Gzgrid, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_Gygrid, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_Gxgrid, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_TxCoilmg, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_TxCoilpe, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_RxCoilx, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_RxCoily, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_b_Mx, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_b_My, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_b_Mz, sizeof(float) * totalSpins));

    // Launch CUDA kernel
    dim3 blockDim(1, 1, 1);
    dim3 gridDim(1, 1, 1);
    size_t sharedMemSize = SeqLen * (5 + 3 * TxCoilNum) * sizeof(float);

    BlochKernelGMGPU<<<gridDim, blockDim, sharedMemSize>>>(
        Gyro,
        d_CS, d_TypeFlag, d_Rho, d_T1, d_T2, d_K, d_Mz, d_My, d_Mx, d_Buffer,
        d_dB0, d_dWRnd, d_Gzgrid, d_Gygrid, d_Gxgrid, d_TxCoilmg, d_TxCoilpe,
        d_RxCoilx, d_RxCoily, d_Sig, RxCoilDefault, TxCoilDefault,
        d_Sx, d_Sy, rfRef, SignalLen, SBufferLen,
        0, // RunMode
        SeqLen,
        d_b_Mz, d_b_My, d_b_Mx,
        SpinMxX, SpinMxY, SpinMxZ, SpinNum, TypeNum, TxCoilNum, RxCoilNum, SeqLen
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Kernel executed successfully!" << std::endl;

    // Free memory
    cudaFree(d_Mx); cudaFree(d_My); cudaFree(d_Mz);
    cudaFree(d_Rho); cudaFree(d_T1); cudaFree(d_T2); cudaFree(d_K);
    cudaFree(d_dB0); cudaFree(d_dWRnd);
    cudaFree(d_CS); cudaFree(d_TypeFlag); cudaFree(d_Buffer); cudaFree(d_Sig);
    cudaFree(d_Sx); cudaFree(d_Sy);
    cudaFree(d_Gzgrid); cudaFree(d_Gygrid); cudaFree(d_Gxgrid);
    cudaFree(d_TxCoilmg); cudaFree(d_TxCoilpe);
    cudaFree(d_RxCoilx); cudaFree(d_RxCoily);
    cudaFree(d_b_Mx); cudaFree(d_b_My); cudaFree(d_b_Mz);

    return 0;
}
