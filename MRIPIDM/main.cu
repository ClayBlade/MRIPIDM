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

float *d_b_Mx, *d_b_My, *d_b_Mz;
float  *d_Buffer;
float *d_K, *d_TypeFlag;
float totalSpins, SBufferLen, SignalLen, SeqLen, RxCoilNum, TxCoilNum, TypeNum, SpinNum, SpinMxZ, SpinMxY, SpinMxX;


int main() {

    float *d_Mz = NULL;
    cudaMalloc( (void**) &d_Mz, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_Mz, Mz, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_My = NULL;
    cudaMalloc( (void**) &d_My, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_My, My, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_Mx = NULL;
    cudaMalloc( (void**) &d_Mx, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_Mx, Mx, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_dWRnd = NULL;
    cudaMalloc( (void**) &d_dWRnd, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_dWRnd, dWRnd, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_Rho = NULL;
    cudaMalloc( (void**) &d_Rho, SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_Rho, Rho, SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_T1 = NULL;
    cudaMalloc( (void**) &d_T1, SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_T1, T1, SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_T2 = NULL;
    cudaMalloc( (void**) &d_T2, SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_T2, T2, SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_Gzgrid = NULL;
    cudaMalloc( (void**) &d_Gzgrid, SpinMxNum * SpinMxSliceNum * sizeof(float)) ;
	cudaMemcpy( d_Gzgrid, Gzgrid, SpinMxNum * SpinMxSliceNum * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_Gygrid = NULL;
    cudaMalloc( (void**) &d_Gygrid, SpinMxNum * SpinMxSliceNum * sizeof(float)) ;
	cudaMemcpy( d_Gygrid, Gygrid, SpinMxNum * SpinMxSliceNum * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_Gxgrid = NULL;
    cudaMalloc( (void**) &d_Gxgrid, SpinMxNum * SpinMxSliceNum * sizeof(float)) ;
	cudaMemcpy( d_Gxgrid, Gxgrid, SpinMxNum * SpinMxSliceNum * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_dB0 = NULL;
    cudaMalloc( (void**) &d_dB0, SpinMxNum * SpinMxSliceNum * sizeof(float)) ;
	cudaMemcpy( d_dB0, dB0, SpinMxNum * SpinMxSliceNum * sizeof(float), cudaMemcpyHostToDevice ) ;

    float *d_TxCoilmg = NULL;
    cudaMalloc( (void**) &d_TxCoilmg, SpinMxNum * SpinMxSliceNum * (*TxCoilNum) * sizeof(float)) ;
	cudaMemcpy( d_TxCoilmg, TxCoilmg, SpinMxNum * SpinMxSliceNum * (*TxCoilNum) * sizeof(float), cudaMemcpyHostToDevice ) ;

    float *d_TxCoilpe = NULL;
    cudaMalloc( (void**) &d_TxCoilpe, SpinMxNum * SpinMxSliceNum * (*TxCoilNum) * sizeof(float)) ;
	cudaMemcpy( d_TxCoilpe, TxCoilpe, SpinMxNum * SpinMxSliceNum * (*TxCoilNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
	
	float *d_RxCoilx = NULL;
    cudaMalloc( (void**) &d_RxCoilx, SpinMxNum * SpinMxSliceNum * (*RxCoilNum) * sizeof(float)) ;
	cudaMemcpy( d_RxCoilx, RxCoilx, SpinMxNum * SpinMxSliceNum * (*RxCoilNum) * sizeof(float), cudaMemcpyHostToDevice ) ;

	float *d_RxCoily = NULL;
    cudaMalloc( (void**) &d_RxCoily, SpinMxNum * SpinMxSliceNum * (*RxCoilNum) * sizeof(float)) ;
	cudaMemcpy( d_RxCoily, RxCoily, SpinMxNum * SpinMxSliceNum * (*RxCoilNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
	
    double *d_CS = NULL;
    cudaMalloc( (void**) &d_CS, *TypeNum * sizeof(double)) ;
	cudaMemcpy( d_CS, CS, *TypeNum * sizeof(double), cudaMemcpyHostToDevice ) ;
	
    /* allocate device memory for GPU execution sequence*/
    float *d_Sig = NULL;
    cudaMalloc( (void**) &d_Sig, (5+3*(*TxCoilNum)) * MaxutsStep * sizeof(float)) ;

    /**/
    float *d_Sx = NULL;
    cudaMalloc( (void**) &d_Sx, SpinMxNum * PreSignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)) ;
    float *d_Sy = NULL;
    cudaMalloc( (void**) &d_Sy, SpinMxNum * PreSignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)) ;

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
