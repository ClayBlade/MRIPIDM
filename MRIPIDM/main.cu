#include <iostream>
#include <cuda_runtime.h>
#include <math.h> 
#include "BlochKernelGMGPU.h" // Your kernel file

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

float *d_b_Mx, *d_b_My, *d_b_Mz, *d_Sy, *d_Sx, *d_Sig, *d_RxCoilx, *d_RxCoily, *d_TxCoilpe, *d_TxCoilmg;
float *d_Gxgrid, *d_Gygrid, *d_Gzgrid, *d_dWRnd, *d_dB0, *d_Buffer;
float *d_Mx, *d_My, *d_Mz, *d_K, *d_T2, *d_T1, *d_Rho, *d_TypeFlag;
float totalSpins, SBufferLen, SignalLen, SeqLen, RxCoilNum, TxCoilNum, TypeNum, SpinNum, SpinMxZ, SpinMxY, SpinMxX;
double *d_CS;

int main() {
    SpinMxX = 0;
    SpinMxY = 0;
    SpinMxZ = 0;
    SpinNum = 12;
    TypeNum = 10;
    TxCoilNum = 125;
    RxCoilNum = 122;
    SeqLen = 25;
    SignalLen = 24;
    SBufferLen = 100;

    totalSpins = SpinMxX * SpinMxY * SpinMxZ * SpinNum * TypeNum;

    CHECK_CUDA(cudaMalloc(&d_Mx, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_My, sizeof(float) * totalSpins));
    CHECK_CUDA(cudaMalloc(&d_Mz, sizeof(float) * totalSpins));

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

    BlochKernelNormalGPU<<<gridDim, blockDim, sharedMemSize>>>(
         Gyro,  d_CS,  d_Rho,  d_T1, d_T2,  d_Mz,  d_My,  d_Mx,
		d_dB0,  d_dWRnd,  d_Gzgrid,  d_Gygrid,  d_Gxgrid,  d_TxCoilmg,  d_TxCoilpe,  d_RxCoilx,  d_RxCoily, 
		d_Sig,  RxCoilDefault,  TxCoilDefault,
		d_Sx,  d_Sy,  rfRef,  SignalLen,  SBufferLen,
		SpinMxX,  SpinMxY,  SpinMxZ,  SpinNum,  TypeNum,  TxCoilNum,  RxCoilNum,  SeqLen
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
