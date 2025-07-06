// BlochKernelGMGPU.h
#ifndef BLOCH_KERNEL_GMGPU_H
#define BLOCH_KERNEL_GMGPU_H

__global__ void BlochKernelNormalGPU(
    float Gyro, double *d_CS, float *d_TypeFlag, float *d_Rho, float *d_T1, float *d_T2, float *d_K,
    float *d_Mz, float *d_My, float *d_Mx, float *d_Buffer, float *d_dB0, float *d_dWRnd,
    float *d_Gzgrid, float *d_Gygrid, float *d_Gxgrid, float *d_TxCoilmg, float *d_TxCoilpe,
    float *d_RxCoilx, float *d_RxCoily, float RxCoilDefault, float TxCoilDefault,
    float *d_Sx, float *d_Sy, float rfRef, int SignalLen, int SBufferLen,
    int RunMode, int utsi, float *d_b_Mz, float *d_b_My, float *d_b_Mx,
    int SpinMxX, int SpinMxY, int SpinMxZ, int SpinNum, int TypeNum, int TxCoilNum, int RxCoilNum, int SeqLen);

#endif
