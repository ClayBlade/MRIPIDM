

/************************************************************************
 MEX code for spin discrete evolution using IPP or Framewave and 
 parallel GPU computation (CUDA) written by Fang Liu (leoliuf@gmail.com).
************************************************************************/

/* system header */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* nVIDIA CUDA header */
#include <cuda.h> 
#include <omp.h>

/* Intel IPP header */
#ifdef IPP
#include <ipps.h>
#include <ippdefs.h>
#endif
/* AMD Framewave header */


/* for fixing error : identifier "IUnknown" is undefined" */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>

#endif

#define PI      3.14159265359 /* pi constant */

#include "helperFuncs.h"
#include "json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json; 


__global__ void BlochKernelNormalGPU(float Gyro, double *d_CS, float *d_Rho, float *d_T1, float *d_T2, float *d_Mz, float *d_My, float *d_Mx,
					 float *d_dB0, float *d_dWRnd, float *d_Gzgrid, float *d_Gygrid, float *d_Gxgrid, float *d_TxCoilmg, float *d_TxCoilpe, float *d_RxCoilx, float *d_RxCoily, 
					 float *d_Sig, float RxCoilDefault, float TxCoilDefault,
					 float *d_Sx, float *d_Sy, float rfRef, int SignalLen, int SBufferLen,
					 int SpinMxX, int SpinMxY, int SpinMxZ, int SpinNum, int TypeNum, int TxCoilNum, int RxCoilNum, int SeqLen)
                     {
                                    /* CUDA index */
                                    unsigned tid	 = blockIdx.x * blockDim.y + threadIdx.y; /* thread id in one slice */
                                    unsigned id      = threadIdx.y;                           /* thread id in one block */

                                    /* sequence buffer in shared memory */
                                    float *g_d_Sig;
                                    extern __shared__ float s_d_Sig[];
                                    int i;
                                    if (SBufferLen !=0){
                                        for (i=0; i< (int)floor((float)(SeqLen*(5 + 3 * TxCoilNum))/(float)blockDim.y); i++){
                                            s_d_Sig[blockDim.y*i+id] = d_Sig[blockDim.y*i+id];
                                        }
                                        if (blockDim.y*i+id < SeqLen*(5 + 3 * TxCoilNum)){
                                            s_d_Sig[blockDim.y*i+id] = d_Sig[blockDim.y*i+id];
                                        }

                                        __syncthreads();
                                        g_d_Sig = s_d_Sig;
                                    }else{
                                        g_d_Sig = d_Sig;
                                    }

                                    /* matrix dim */
                                    int SpinMxNum	 = SpinMxX * SpinMxY;
                                    int SpinMxAllNum = SpinMxX * SpinMxY * SpinMxZ;
                                    
                                    /* signal counter*/
                                    int Signalptr;
                                    
                                    /* dt buffer */
                                    float dt;
                                    float ExpdtT2;
                                    float ExpdtT1;
                                    float M0dtT1;
                                    
                                    /* matrix pointers */
                                    float *p_d_Mz;
                                    float *p_d_My;
                                    float *p_d_Mx;
                                    float *p_d_dWRnd;
                                    float *p_d_Rho;
                                    float *p_d_T1;
                                    float *p_d_T2;
                                    float *p_d_Gzgrid;
                                    float *p_d_Gygrid;
                                    float *p_d_Gxgrid;
                                    float *p_d_dB0;
                                    float *p_d_TxCoilmg;
                                    float *p_d_TxCoilpe;
                                    float *p_d_RxCoilx;
                                    float *p_d_RxCoily;
                                    float *p_d_Sx;
                                    float *p_d_Sy;
                                    
                                    float *p_d_rfAmp;
                                    float *p_d_rfPhase;
                                    float *p_d_rfFreq;
                                    float *p_d_GzAmp;
                                    float *p_d_GyAmp;
                                    float *p_d_GxAmp;
                                    float *p_d_dt;
                                    float *p_d_ADC;
                                    
                                    /* multi-Tx  variables */
                                    float rfAmpSum; 
                                    float rfAmp;
                                    float rfPhase;
                                    float rfFreq;
                                    float buffer1;
                                    float buffer2;
                                    float buffer3;
                                    float buffer4;
                                    
                                    /* spin variables */
                                    float Mx, My, Mz;
                                    float T1, T2, Rho, dWRnd;
                                    float Gzgrid, Gygrid, Gxgrid, dB0;
                                    
                                    /* temporary  variables */
                                    float dW, sinAlpha, sinBeta, sinPhi, cosAlpha, cosBeta, cosPhi, Alpha, Beta;
                                    float bufferMz, bufferMy, bufferMx;
                                    
                                    /* loop through slice <- spins <- species */
                                    for (int t=0; t < TypeNum; t++){
                                        for (int s=0; s < SpinNum; s++){
                                            for (int k=0; k < SpinMxZ; k++){
                                                p_d_Rho			= d_Rho 		+ k * SpinMxNum	+ tid 	+ t * SpinMxAllNum;
                                                p_d_T1        	= d_T1 			+ k * SpinMxNum	+ tid 	+ t * SpinMxAllNum;
                                                p_d_T2        	= d_T2 			+ k * SpinMxNum	+ tid 	+ t * SpinMxAllNum;
                                                
                                                if (*p_d_T2==0 || *p_d_T1==0 || *p_d_Rho==0) continue; /* avoid background  23%*/
                                                
                                                p_d_Mz        	= d_Mz 			+ k * SpinMxNum	+ tid 	+ t * (SpinMxAllNum * SpinNum) 	+ s * SpinMxAllNum;
                                                p_d_My        	= d_My 			+ k * SpinMxNum	+ tid 	+ t * (SpinMxAllNum * SpinNum) 	+ s * SpinMxAllNum;
                                                p_d_Mx        	= d_Mx 			+ k * SpinMxNum	+ tid 	+ t * (SpinMxAllNum * SpinNum) 	+ s * SpinMxAllNum;
                                                p_d_dWRnd		= d_dWRnd 		+ k * SpinMxNum	+ tid 	+ t * (SpinMxAllNum * SpinNum) 	+ s * SpinMxAllNum;
                                                p_d_Gzgrid    	= d_Gzgrid 		+ k * SpinMxNum	+ tid ;
                                                p_d_Gygrid    	= d_Gygrid 		+ k * SpinMxNum	+ tid ;
                                                p_d_Gxgrid    	= d_Gxgrid 		+ k * SpinMxNum	+ tid ;
                                                p_d_dB0       	= d_dB0 		+ k * SpinMxNum	+ tid ;
                                                p_d_TxCoilmg	= d_TxCoilmg	+ k * SpinMxNum	+ tid ;
                                                p_d_TxCoilpe	= d_TxCoilpe 	+ k * SpinMxNum	+ tid ;
                                                p_d_RxCoilx		= d_RxCoilx		+ k * SpinMxNum	+ tid ;
                                                p_d_RxCoily		= d_RxCoily 	+ k * SpinMxNum	+ tid ;
                                                
                                                Mx              = *p_d_Mx;
                                                My				= *p_d_My;
                                                Mz              = *p_d_Mz;
                                                T1              = *p_d_T1;
                                                T2 				= *p_d_T2;
                                                Rho 			= *p_d_Rho;
                                                dWRnd			= *p_d_dWRnd;
                                                Gzgrid			= *p_d_Gzgrid;
                                                Gygrid			= *p_d_Gygrid;
                                                Gxgrid			= *p_d_Gxgrid;
                                                dB0				= *p_d_dB0;
                                                
                                                Signalptr = 0;
                                                dt		  = 0;
                                                for (int q=0; q< SeqLen; q++){
                                                    p_d_dt		= g_d_Sig + q * (5 + 3 * TxCoilNum);
                                                    p_d_rfAmp 	= g_d_Sig + q * (5 + 3 * TxCoilNum) + 1;
                                                    p_d_rfPhase = g_d_Sig + q * (5 + 3 * TxCoilNum) + 2;
                                                    p_d_rfFreq 	= g_d_Sig + q * (5 + 3 * TxCoilNum) + 3;
                                                    p_d_GzAmp 	= g_d_Sig + q * (5 + 3 * TxCoilNum) + 3 * TxCoilNum + 1;
                                                    p_d_GyAmp 	= g_d_Sig + q * (5 + 3 * TxCoilNum) + 3 * TxCoilNum + 2;
                                                    p_d_GxAmp 	= g_d_Sig + q * (5 + 3 * TxCoilNum) + 3 * TxCoilNum + 3;
                                                    p_d_ADC 	= g_d_Sig + q * (5 + 3 * TxCoilNum) + 3 * TxCoilNum + 4;
                                                    
                                                    /* signal acquisition */
                                                    if (*p_d_ADC == 1) {
                                                        for (int c = 0; c < RxCoilNum; c++){  /* signal acquisition per Rx coil */    
                                                            /* RxCoil sensitivity */
                                                            if (RxCoilDefault ==0){
                                                                buffer1 =  Mx * (* (p_d_RxCoilx + c * SpinMxAllNum))
                                                                        +My * (* (p_d_RxCoily + c * SpinMxAllNum));
                                                                buffer2 = -Mx * (* (p_d_RxCoily + c * SpinMxAllNum))
                                                                        +My * (* (p_d_RxCoilx + c * SpinMxAllNum));
                                                                buffer3 = buffer1;
                                                                buffer4 = buffer2;
                                                            }else{
                                                                buffer1 = Mx;
                                                                buffer2 = My;
                                                                buffer3 = buffer1;
                                                                buffer4 = buffer2;
                                                            }
                                                            
                                                            /* rfRef for demodulating rf Phase */
                                                            if (rfRef!=0){
                                                                buffer1 = cos(-rfRef) * buffer1;
                                                                buffer2 = -sin(-rfRef) * buffer2;
                                                                buffer3 = sin(-rfRef) * buffer3;
                                                                buffer4 = cos(-rfRef) * buffer4;
                                                                buffer1 = buffer1 + buffer2;
                                                                buffer3 = buffer3 + buffer4;
                                                            }else{
                                                                buffer3 = buffer4;
                                                            }
                                                            
                                                            /* signal buffer pointer */
                                                            p_d_Sx = d_Sx + tid + t * (SpinMxNum * SignalLen * RxCoilNum) + c * (SpinMxNum * SignalLen) + Signalptr * SpinMxNum;
                                                            p_d_Sy = d_Sy + tid + t * (SpinMxNum * SignalLen * RxCoilNum) + c * (SpinMxNum * SignalLen) + Signalptr * SpinMxNum;
                                                            
                                                            /* update signal buffer */
                                                            *p_d_Sx += buffer1;
                                                            *p_d_Sy += buffer3;
                                                        }
                                                        Signalptr++;
                                                    }
                                                    
                                                    /* spin precession */
                                                    dW =    dB0 * Gyro + dWRnd + 2 * PI * (float)d_CS[t]
                                                            + Gzgrid * (*p_d_GzAmp) * Gyro
                                                            + Gygrid * (*p_d_GyAmp) * Gyro
                                                            + Gxgrid * (*p_d_GxAmp) * Gyro;
                                                    
                                                    rfAmpSum = 0;
                                                    for (int c = 0; c<TxCoilNum; c++){
                                                        rfAmpSum+=fabs(p_d_rfAmp[c*3]);
                                                    }

                                                    if (rfAmpSum != 0){
                                                        if (TxCoilNum == 1) { /* single-Tx */
                                                            rfAmp   = p_d_rfAmp[0];
                                                            rfPhase = p_d_rfPhase[0];
                                                            rfFreq  = p_d_rfFreq[0]; /* note rfFreq is defined as fB0-frf */

                                                            dW		+= 2 * PI * rfFreq;
                                                            buffer1	 = *p_d_TxCoilmg * rfAmp;
                                                            buffer2	 = *p_d_TxCoilpe + rfPhase;

                                                            Alpha	 = sqrt(pow(dW,2) + pow(buffer1,2) * pow(Gyro,2)) * (*p_d_dt);  /* calculate alpha */
                                                            Beta	 = atan(dW/(buffer1 * Gyro));  /* calculate beta */
                                                            sinAlpha = sin(Alpha);
                                                            sinBeta	 = sin(Beta);
                                                            cosAlpha = cos(Alpha);
                                                            cosBeta  = cos(Beta);
                                                            cosPhi   = cos(-buffer2);
                                                            sinPhi   = sin(-buffer2);
                                                        }
                                                        else{
                                                            buffer3 = 0;
                                                            buffer4 = 0;
                                                            for (int c = 0; c<TxCoilNum; c++){ /* multi-Tx,  sum all (B1+ * rf) */
                                                                rfAmp   = p_d_rfAmp[c*3];
                                                                rfPhase = p_d_rfPhase[c*3];
                                                                rfFreq  = p_d_rfFreq[c*3]; /* note rfFreq is defined as fB0-frf */
                                                                if (rfAmp !=0 ){
                                                                    dW      += 2 * PI * rfFreq;
                                                                    buffer1  = *(p_d_TxCoilmg + c * SpinMxAllNum) * rfAmp;
                                                                    buffer2  = *(p_d_TxCoilpe + c * SpinMxAllNum) + rfPhase;
                                                                    buffer3 += buffer1 * cos(buffer2);
                                                                    buffer4 += buffer1 * sin(buffer2);
                                                                }
                                                            }
                                                            buffer1 = sqrt(pow(buffer3, 2) + pow(buffer4,2));
                                                            buffer2 = atan2(buffer4, buffer3);

                                                            Alpha	  = sqrt(pow(dW,2) + pow(buffer1,2) * pow(Gyro,2)) * (*p_d_dt);  /* calculate alpha */
                                                            Beta      = atan(dW/(buffer1 * Gyro));  /* calculate beta */
                                                            sinAlpha  = sin(Alpha);
                                                            sinBeta   = sin(Beta);
                                                            cosAlpha  = cos(Alpha);
                                                            cosBeta   = cos(Beta);
                                                            cosPhi    = cos(-buffer2);
                                                            sinPhi    = sin(-buffer2);
                                                        }

                                                        buffer1 = pow(cosBeta,2)*cosPhi - sinBeta*(sinAlpha*sinPhi - cosAlpha*cosPhi*sinBeta);
                                                        buffer2 = sinPhi*pow(cosBeta,2) + sinBeta*(cosPhi*sinAlpha + cosAlpha*sinBeta*sinPhi);
                                                        
                                                        bufferMx = Mx * (cosPhi*buffer1 + sinPhi*(cosAlpha*sinPhi + cosPhi*sinAlpha*sinBeta))
                                                                -My * (sinPhi*buffer1 - cosPhi*(cosAlpha*sinPhi + cosPhi*sinAlpha*sinBeta))
                                                                +Mz * (cosBeta*(sinAlpha*sinPhi - cosAlpha*cosPhi*sinBeta) + cosBeta*cosPhi*sinBeta);  /*Calculate Mx */
                                                        
                                                        bufferMy = My * (sinPhi*buffer2 + cosPhi*(cosAlpha*cosPhi - sinAlpha*sinBeta*sinPhi))
                                                                -Mx * (cosPhi*buffer2 - sinPhi*(cosAlpha*cosPhi - sinAlpha*sinBeta*sinPhi))
                                                                +Mz * (cosBeta*(cosPhi*sinAlpha + cosAlpha*sinBeta*sinPhi) - cosBeta*sinBeta*sinPhi);  /*Calculate My */
                                                        
                                                        bufferMz = Mx * (cosPhi*(cosBeta*sinBeta - cosAlpha*cosBeta*sinBeta) - cosBeta*sinAlpha*sinPhi)
                                                                -My * (sinPhi*(cosBeta*sinBeta - cosAlpha*cosBeta*sinBeta) + cosBeta*cosPhi*sinAlpha)
                                                                +Mz * (cosAlpha*pow(cosBeta,2) + pow(sinBeta,2));     /*Calculate Mz */
                                                    }
                                                    else{

                                                        Alpha	 = dW * (*p_d_dt);  /* calculate alpha */
                                                        sinAlpha = sin(Alpha);
                                                        cosAlpha = cos(Alpha);

                                                        bufferMx = Mx * cosAlpha + My * sinAlpha;     /* calculate Mx */
                                                        bufferMy = My * cosAlpha - Mx * sinAlpha;     /* calculate My */
                                                        bufferMz = Mz ;								  /* calculate Mz */
                                                    }
                                                    
                                                    /* relax */
                                                    if (dt != *p_d_dt){ /* exp & division is very time consuming */
                                                        ExpdtT2 = exp(-(*p_d_dt)/(T2));
                                                        ExpdtT1 = exp(-(*p_d_dt)/(T1));
                                                        M0dtT1  = (Rho*(ExpdtT1 - 1))/SpinNum;
                                                        dt		= *p_d_dt;
                                                    }
                                                    Mx = bufferMx * ExpdtT2;
                                                    My = bufferMy * ExpdtT2;
                                                    Mz = bufferMz * ExpdtT1 - M0dtT1;
                                                }
                                                
                                                *p_d_Mx  = Mx;
                                                *p_d_My  = My;
                                                *p_d_Mz  = Mz;
                                            }
                                        }
                                    }
                                    //printf("Mx: %f, My: %f, Mz: %f\n", Mx, My, Mz);
                                }

int main(){
    std::ifstream inputFile("/root/output/labeledSpaceJSON/1.pkl.json");

    json data_obj;
    inputFile >> data_obj; 

/* pointers for VObj */
    int SpinMxNum, SpinMxColNum, SpinMxRowNum, SpinMxSliceNum, SpinMxDimNum;



/* pointers for VMag */
    float *dB0, *dWRnd, *Gzgrid, *Gygrid, *Gxgrid;
    
/* pointers for VCoi */
    float *RxCoilx, *RxCoily, *TxCoilmg, *TxCoilpe;
	double *RxCoilDefault, *TxCoilDefault;
    
/* pointers for VCtl */
    double *CS;
    int *TRNum, *MaxThreadNum, ThreadNum;
	int *ActiveThreadNum;
	int *GPUIndex;
    
/* pointers for VSeq */
    double *utsLine, *tsLine, *rfAmpLine, *rfPhaseLine, *rfFreqLine, *rfCoilLine, *GzAmpLine, *GyAmpLine, *GxAmpLine, *ADCLine, *ExtLine, *flagsLine;

/* pointers for VVar */
    double *t, *dt, *rfAmp, *rfPhase, *rfFreq, *rfCoil, *rfRef, *GzAmp, *GyAmp, *GxAmp, *ADC, *Ext, *KzTmp, *KyTmp, *KxTmp, *gpuFetch;
    int *utsi, *rfi, *Gzi, *Gyi, *Gxi, *ADCi, *Exti, *TRCount;
    
/* pointers for VSig */
    double *Sx, *Sy, *Kz, *Ky, *Kx;
	double *p_Sx, *p_Sy;
	
/* loop control */
    int i=0, j=0, s=0, Signali=0, Signalptr=0, PreSignalLen=1, SignalLen=1, SBufferLen=0, Typei, RxCoili, TxCoili;
    int MaxStep, MaxutsStep, MaxrfStep, MaxGzStep, MaxGyStep, MaxGxStep, *SpinNum, *TxCoilNum, *RxCoilNum, *SignalNum;
    double flag[6];
    
/* IPP or FW buffer */
    float buffer, *Sxbuffer, *Sybuffer;
	
/* function status */
    int ExtCall;
    
/* GPU execution sequence */
	std::vector<float> g_Sig;	

/* assign pointers */
    /*VObj*/
    double* Gyro             = new double;
        *Gyro = 2.67e08; /* Gyromagnetic ratio for Hydrogen in Hz/T */
    int* TypeNum             = new int;
        *TypeNum = 1;
    SpinNum         = new int;
        *SpinNum         =   1;
        std::vector<float> Mz = data_obj["Mz"].get<std::vector<float>>();
        std::vector<float> My = data_obj["My"].get<std::vector<float>>();
        std::vector<float> Mx = data_obj["Mx"].get<std::vector<float>>();
        std::vector<float> Rho = data_obj["Rho"].get<std::vector<float>>();
        std::vector<float> T1 = data_obj["T1"].get<std::vector<float>>();
        std::vector<float> T2 = data_obj["T2"].get<std::vector<float>>();


	GPUIndex		= new int;
    *GPUIndex       = 0; /* default value */


/* get size of spin matrix */
    SpinMxDimNum    		= 3;
    size_t* SpinMxDims = (size_t*) malloc(SpinMxDimNum * sizeof(size_t));
    SpinMxDims[0] = (int)data_obj["xSize"];
    SpinMxDims[1] = (int)data_obj["ySize"]; 
    SpinMxDims[2] = (int)data_obj["zSize"];
    std::cout << "SpinMxDims: " << SpinMxDims[0] << " " << SpinMxDims[1] << " " << SpinMxDims[2] << std::endl;

    /*Might be y by x by z*/
	
    SpinMxRowNum    		= SpinMxDims[0];
    SpinMxColNum    		= SpinMxDims[1];
    SpinMxNum       		= SpinMxDims[0] * SpinMxDims[1];
    if (SpinMxDimNum == 2){
        SpinMxSliceNum = 1;
    }else{
        SpinMxSliceNum = SpinMxDims[2];
    }
	
/* choose selected GPU */
	if( cudaSuccess != cudaSetDevice(*GPUIndex)){
        return 0;
    }
    std::cout << "Using GPU: " << *GPUIndex << std::endl;
	
/* set GPU grid & block configuration*/
    cudaDeviceProp deviceProp;
    memset( &deviceProp, 0, sizeof(deviceProp));
    if( cudaSuccess != cudaGetDeviceProperties(&deviceProp, *GPUIndex)){
        return 0;
    }

	dim3 dimGridImg(SpinMxColNum,1,1);
    dim3 dimBlockImg(1,SpinMxRowNum,1);

	for (i=SpinMxColNum - 1; i >= deviceProp.multiProcessorCount; i--){
		if ( SpinMxNum % i == 0 ){
			if (SpinMxNum/i > deviceProp.maxThreadsPerBlock) break;
			if ((SpinMxNum/i)*63 > deviceProp.regsPerBlock) break; // 63 registers per thread for current kernel
			dimGridImg.x = i;
		    dimBlockImg.y = SpinMxNum/i;
		}
	}
	i=0;

    /*VCtl*/
    CS              = new double; 
    *CS = 0.0; /* default value */
    TRNum  			= new int;
    *TRNum          = 10; /* default value */
    MaxThreadNum    = new int;
    *MaxThreadNum   = deviceProp.maxThreadsPerBlock;
	ActiveThreadNum = new int;
    *ActiveThreadNum = *MaxThreadNum; /* default value */
    std::cout << "MaxThreadNum: " << *MaxThreadNum << std::endl;
    std::cout << "SpinMxNum: " << SpinMxNum << std::endl;  
    std::cout << "SpinMxSliceNum: " << SpinMxSliceNum << std::endl;
    std::cout << "TypeNum: " << *TypeNum << std::endl;

    
    /*VMag*/
    dB0 = new float[SpinMxNum * SpinMxSliceNum]; /*Initialized as 0*/
    dWRnd = new float[SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum)]; 
    Gzgrid = new float[SpinMxNum * SpinMxSliceNum];
    Gygrid = new float[SpinMxNum * SpinMxSliceNum];
    Gxgrid = new float[SpinMxNum * SpinMxSliceNum];

    std::cout << SpinMxNum * SpinMxSliceNum << " spins in total." << std::endl;
    
    /*VCoi*/
    TxCoilNum       = new int;
    *TxCoilNum      = 1; /* default value */
    RxCoilNum       = new int;
    *RxCoilNum      = 1; /* default value */
	TxCoilDefault   = new double;
    *TxCoilDefault = 1.0;
    RxCoilDefault   = new double;
    *RxCoilDefault = 1.0;
    TxCoilmg        = new float[SpinMxNum * SpinMxSliceNum * (*TxCoilNum)];
    TxCoilpe        = new float[SpinMxNum * SpinMxSliceNum * (*TxCoilNum)];
    RxCoilx         = new float[SpinMxNum * SpinMxSliceNum * (*RxCoilNum)];
    RxCoily         = new float[SpinMxNum * SpinMxSliceNum * (*RxCoilNum)];
    
     

    MaxStep         = 2500;
    MaxutsStep      = MaxStep * 10;
    MaxrfStep       = 2500; /*Just fill in the zeros*/
    MaxGzStep       = 2500;
    MaxGyStep       = 2500;
    MaxGxStep       = 2500;

    /*VSeq*/
    utsLine         = new double[MaxutsStep];
    tsLine          = new double[MaxStep * 6];
    rfAmpLine       = new double[MaxrfStep];
    rfPhaseLine     = new double[MaxrfStep];
    rfFreqLine      = new double[MaxrfStep];
    rfCoilLine      = new double[MaxrfStep];
    GzAmpLine       = new double[MaxGzStep];
    GyAmpLine       = new double[MaxGyStep];
    GxAmpLine       = new double[MaxGxStep];
    ADCLine         = new double[MaxStep];
    ExtLine         = new double[MaxStep];
    flagsLine       = new double[MaxStep * 6];
    
    

	
    /*VVar*/
	t               = new double;
    *t              =  0;
    dt              = new double;
    *dt             = 20e-6; /* 20 us */
    rfAmp           = new double;
    *rfAmp          =  0;
    rfPhase         = new double;
    *rfPhase        =  0;
    rfFreq          = new double;
    *rfFreq         =  0;
    rfCoil          = new double;
    *rfCoil         =  1;
    rfRef           = new double;
    *rfRef          =  0;
    GzAmp           = new double;
    *GzAmp           =  0;
    GyAmp           = new double;
    *GyAmp           =  0;
    GxAmp           = new double;
    *GxAmp           =  0;
    ADC             = new double;
    *ADC            =  0;
    Ext             = new double;
    *Ext            =  1;
    KzTmp           = new double;
    *KzTmp          = 0;
    KyTmp           = new double;
    *KyTmp          =  0;
    KxTmp           = new double;
    *KxTmp          = 0;
	gpuFetch     	= new double;
    *gpuFetch       = 1;
    utsi            = new int;
    *utsi           =  0;
    rfi             = new int;
    *rfi            =  0;
    Gzi             = new int;
    *Gzi            =  0;
    Gyi             = new int;
    *Gyi            =  0;
    Gxi             = new int;
    *Gxi            = 0;
    ADCi            = new int;
    *ADCi           =  0;
    Exti            = new int;
    *Exti           =  0;
    TRCount         = new int;
    *TRCount        =  0;
	
    /*VSig*/ 
	Sy              = new double[SpinMxNum * PreSignalLen * (*TypeNum) * (*RxCoilNum)];
    Sx              = new double[SpinMxNum * PreSignalLen * (*TypeNum) * (*RxCoilNum)];
    
    Kz              = new double;
    *Kz              = 0;
    Ky              = new double;
    *Ky              = 0;
    Kx              = new double;
    *Kx              = 0;
    SignalNum       = new int;
    *SignalNum      = 0;

    /*Initialize Arrays: dB0, dWRn, Gzgrid, Gygrid, Gxgrid
    TxCoilmg, TxCoilpe, RxCoilx, RxCoily, TxCoilNum,
    CS,
    utsLine, tsLine, rfAmpLine, rfPhaseLine, rfFreqLine, rfCoilLine, GzAmpLine, GyAmpLine, GxAmpLine, ADCLine, ExtLine, flagsLine  
    Sy, Sx, Kx, Ky, Kz
    */
    int xSiz = (int)data_obj["xSize"];
    int ySiz = (int)data_obj["ySize"];


   for (int x = 0; x < xSiz; x++){
    for (int y = 0; y < ySiz; y++){
    for (int b = 0; b < SpinMxSliceNum; b++){
            dB0[b * xSiz * ySiz + y * xSiz + x] = 0.0;
            Gzgrid[b * xSiz * ySiz + y * xSiz + x] = ((b-SpinMxSliceNum)/2) * 0.25/SpinMxSliceNum; /*0.2/size*/
            Gygrid[b * xSiz * ySiz + y * xSiz + x] = (y-ySiz/2) * 0.25/ySiz; /*0.2/size*/
            Gxgrid[b * xSiz * ySiz + y * xSiz + x] = (x-xSiz/2) * 0.25/xSiz; /*0.2/size*/

            for (int d = 0; d < *SpinNum; d++){
                int idx = d * SpinMxSliceNum * xSiz * ySiz + b * xSiz * ySiz + y * xSiz + x;
                dWRnd[idx] = 0;
                //std::cout << "dWRnd[" << idx << "] = " << dWRnd[idx] << std::endl;
            }

            for (int e = 0; e < *TxCoilNum; e++){
                int idx = e * SpinMxSliceNum * xSiz * ySiz + b * xSiz * ySiz + y * xSiz + x;
                TxCoilmg[idx] = 0.0;
                TxCoilpe[idx] = 0.0;
                //std::cout << "TxCoilmg[" << idx << "] = " << TxCoilmg[idx] << std::endl;
            }

            for (int f = 0; f < *RxCoilNum; f++){
                int idx = f * SpinMxSliceNum * xSiz * ySiz + b * xSiz * ySiz + y * xSiz + x;
                RxCoilx[idx] = 0.0;
                RxCoily[idx] = 0.0;
                //std::cout << "RxCoilx[" << idx << "] = " << RxCoilx[idx] << std::endl;
            }
            
     }
    }
   }


 /*Initialize Sequence */  

for (int i = 0; i < MaxStep; i++){
    if (i <= 128){ 
        rfAmpLine[i] = sin(i*(3*3.14f)/128)/i;
        rfPhaseLine[i] = 3.14f;  
        rfFreqLine[i] = 1;
        rfCoilLine[i] = 1;
        GzAmpLine[i] = 1;
        GyAmpLine[i] = 0;
        GxAmpLine[i] = 0;
        ADCLine[i] = 0;
        ExtLine[i] = 1;
    }
    if (i >= 320 && i >= 192){
        rfAmpLine[i] = sin((3*3.14f*(i-64))/128)/i-64;
        rfPhaseLine[i] = 3.14f;  
        rfFreqLine[i] = 1;
        rfCoilLine[i] = 1;
        GzAmpLine[i] = 1;
        GyAmpLine[i] = 0;
        GxAmpLine[i] = 0;
        ADCLine[i] = 0;
        ExtLine[i] = 1;
        if (i >= 300){
            GxAmpLine[i] = 1;
            GyAmpLine[i] = 1;
        }
    }
    if (i >= 324){
        rfAmpLine[i] = 0;
        rfPhaseLine[i] = 0;  
        rfFreqLine[i] = 0;
        rfCoilLine[i] = 0;
        GzAmpLine[i] = 1;
        GyAmpLine[i] = 1;
        GxAmpLine[i] = 1;
        ADCLine[i] = 1;
        ExtLine[i] = 1;
    }

    for (int j = 0; j < 10; j++){
        utsLine[i * 10 + j] = *dt * i + j * 0.1f; // Just an example, adjust as needed
    }
    for (int j = 0; j < 6; j++){
        tsLine[i * 6 + j] = *dt * i;
        flagsLine[i * 6] = rfAmpLine[i];
        flagsLine[i * 6 + 1] = GzAmpLine[i]; 
        flagsLine[i * 6 + 2] = GyAmpLine[i];
        flagsLine[i * 6 + 3] = GxAmpLine[i];
        flagsLine[i * 6 + 4] = ADCLine[i];
        flagsLine[i * 6 + 5] = ExtLine[i];

    }    
}


    
    
	
/* allocate device memory for matrices */
    float *d_Mz = NULL;
    cudaMalloc( (void**) &d_Mz, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_Mz, Mz.data(), SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_My = NULL;
    cudaMalloc( (void**) &d_My, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_My, My.data(), SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_Mx = NULL;
    cudaMalloc( (void**) &d_Mx, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_Mx, Mx.data(), SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_dWRnd = NULL;
    cudaMalloc( (void**) &d_dWRnd, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_dWRnd, dWRnd, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_Rho = NULL;
    cudaMalloc( (void**) &d_Rho, SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_Rho, Rho.data(), SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_T1 = NULL;
    cudaMalloc( (void**) &d_T1, SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_T1, T1.data(), SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
    float *d_T2 = NULL;
    cudaMalloc( (void**) &d_T2, SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float)) ;
	cudaMemcpy( d_T2, T2.data(), SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice ) ;
    
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
	
/* set CPU signal buffer */
	Sxbuffer    = ippsMalloc_32fHF(SpinMxNum * PreSignalLen * (*TypeNum) * (*RxCoilNum));
	Sybuffer    = ippsMalloc_32fHF(SpinMxNum * PreSignalLen * (*TypeNum) * (*RxCoilNum));

/* allocate device memory for buffering acquired signal */
    float *d_Sx = NULL;
    cudaMalloc( (void**) &d_Sx, SpinMxNum * PreSignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)) ;
    float *d_Sy = NULL;
    cudaMalloc( (void**) &d_Sy, SpinMxNum * PreSignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)) ;

/* start simulator execution loop */
    while (i < MaxStep){
        //std::cout << "Processing step: " << i << std::endl;
        /* check MR sequence pulse flag */
        flag[0]=0;
        flag[1]=0;
        flag[2]=0;
        flag[3]=0;
        flag[4]=0;
        flag[5]=0;
        if (tsLine[i]!=tsLine[i+1]){
            flag[0]+=flagsLine[i*6];
            flag[1]+=flagsLine[i*6+1];
            flag[2]+=flagsLine[i*6+2];
            flag[3]+=flagsLine[i*6+3];
            flag[4]+=flagsLine[i*6+4];
            flag[5]+=flagsLine[i*6+5];
            i++;
        }
        else{
            flag[0]+=flagsLine[i*6];
            flag[1]+=flagsLine[i*6+1];
            flag[2]+=flagsLine[i*6+2];
            flag[3]+=flagsLine[i*6+3];
            flag[4]+=flagsLine[i*6+4];
            flag[5]+=flagsLine[i*6+5];
            
            while (tsLine[i]==tsLine[i+1]){
                flag[0]+=flagsLine[(i+1)*6];
                flag[1]+=flagsLine[(i+1)*6+1];
                flag[2]+=flagsLine[(i+1)*6+2];
                flag[3]+=flagsLine[(i+1)*6+3];
                flag[4]+=flagsLine[(i+1)*6+4];
                flag[5]+=flagsLine[(i+1)*6+5];
                i++;
                if (i==MaxStep-1){
                    break;
                }
            }
            i++;
        }

        /*tsLine contains timeline of if events happen at this timestep or not for event event
        */


        //std::cout << "checkpoint1" << std::endl;
        /* update pulse status */
        *t 	= *(utsLine + *utsi);
        *dt 	= *(utsLine + (int)min(*utsi+1, MaxutsStep-1))-*(utsLine + *utsi);
        *utsi = (int)min(*utsi+1, MaxutsStep-1);
		if (*dt > 0) g_Sig.push_back((float)*dt);
		
        if (flag[0]>=1 ){ /* update rfAmp, rfPhase, rfFreq, rfCoil for multiple rf lines */
            for (j = 0; j < flag[0]; j++){
				 *rfCoil = *(rfCoilLine+ *rfi);
				 TxCoili = (int)(*rfCoil);
				 s = *rfi + 1;
				 while (s < MaxrfStep){
					if (*rfCoil == *(rfCoilLine + s)){
						if (fabs(*(rfAmpLine+ *rfi)) <= fabs(*(rfAmpLine + s)))
							*(rfAmp + TxCoili - 1)= *(rfAmpLine+ *rfi);
						else
							*(rfAmp + TxCoili - 1)= *(rfAmpLine+ s);
	                    
						if (fabs(*(rfPhaseLine+ *rfi)) <= fabs(*(rfPhaseLine + s)))
							*(rfPhase + TxCoili - 1)= *(rfPhaseLine+ *rfi);
						else
							*(rfPhase + TxCoili - 1)= *(rfPhaseLine+ s);
	                    
						if (fabs(*(rfFreqLine+ *rfi)) <= fabs(*(rfFreqLine + s)))
							*(rfFreq + TxCoili - 1)= *(rfFreqLine+ *rfi);
						else
							*(rfFreq + TxCoili - 1)= *(rfFreqLine+ s);
						break;
					}
					s++;
				 }
				 (*rfi)++;
            }
			
			for (j = 0; j < *TxCoilNum; j++){ /* multi-Tx, deal with rfPhase */
				if (rfAmp[j]<0){
					rfAmp[j]=fabs(rfAmp[j]);
					rfPhase[j]=rfPhase[j]+PI;
				}
			}
			
        }
        //std::cout << "checkpoint2" << std::endl;
		if (*dt > 0){
			for (j = 0; j < *TxCoilNum; j++){
				g_Sig.push_back((float)rfAmp[j]);
				g_Sig.push_back((float)rfPhase[j]);
				g_Sig.push_back((float)rfFreq[j]);
			}
		}
        //std::cout << "checkpoint2.1" << std::endl;
        if (flag[1]==1 ){ /* update GzAmp */
            if (fabs(*(GzAmpLine+ *Gzi)) <= fabs(*(GzAmpLine + (int)min((*Gzi)+1, MaxGzStep-1))))
                *GzAmp = *(GzAmpLine+ *Gzi);
            else
                *GzAmp = *(GzAmpLine+ (*Gzi)+1);
            
            (*Gzi)++;
        }
        if (*dt > 0) g_Sig.push_back((float)*GzAmp);
		
        //std::cout << "checkpoint2.2" << std::endl;

        if (flag[2]==1 ){ /* update GyAmp */
            if (fabs(*(GyAmpLine + *Gyi)) <= fabs(*(GyAmpLine + (int)min(*Gyi+1, MaxGyStep-1))))
                *GyAmp = *(GyAmpLine+ *Gyi);
            else
                *GyAmp = *(GyAmpLine+ *Gyi+1);
            
            (*Gyi)++;
        }
        if (*dt > 0) g_Sig.push_back((float)*GyAmp);
		
        //std::cout << "checkpoint2.3" << std::endl;

        if (flag[3]==1 ){ /* update GxAmp */
            if (fabs(*(GxAmpLine+ *Gxi)) <= fabs(*(GxAmpLine + (int)min(*Gxi+1, MaxGxStep-1))))
                *GxAmp = *(GxAmpLine+ *Gxi);
            else
                *GxAmp = *(GxAmpLine+ *Gxi+1);
            
            (*Gxi)++;
        }			
		if (*dt > 0) g_Sig.push_back((float)*GxAmp);
        
        //std::cout << "checkpoint3" << std::endl;

        *ADC = 0;   /* avoid ADC overflow */
        if (flag[4]==1){ /* update ADC */
            *ADC = *(ADCLine+ *ADCi);
            (*ADCi)++;
        }
		if (*dt > 0) g_Sig.push_back((float)*ADC);
		
		if (*ADC == 1){
			/* update k-space */
            Kz[Signali] += *KzTmp;
            Ky[Signali] += *KyTmp;
            Kx[Signali] += *KxTmp;
            Signali++;
		}
		
		 /* update Kz, Ky & Kx buffer */
        *KzTmp +=(*GzAmp)*(*dt)*(*Gyro/(2*PI));
        *KyTmp +=(*GyAmp)*(*dt)*(*Gyro/(2*PI));
        *KxTmp +=(*GxAmp)*(*dt)*(*Gyro/(2*PI));
		

        //if (flag[5] != 0){ std::cout << "flag passed" << std::endl; }
        //if (*Ext !=0){std::cout << "ext" << *Ext << std::endl;}

        

        if (flag[5]!=0){ /* update Ext */         //changed from ==1

            *Ext = *(ExtLine+ *Exti);
            /* execute extended process */
            //std::cout << "Passed flag" << std::endl;
            //std::cout << "ext: " << *Ext << std::endl;
            if (*Ext != 0){
                //std::cout << "Passed Ext" << std::endl;
                //std::cout << "g.sig: " << g_Sig.size() << std::endl;
				if (g_Sig.size() !=0){
                    //std::cout << "passed g_sig" << std::endl;
				
					/* calculate signal length */
					SignalLen = Signali-Signalptr;

					/* reset buffer */
					if (PreSignalLen!=SignalLen && SignalLen>0){
						PreSignalLen = SignalLen;
						/* allocate device memory for acquired signal buffer */
						cudaFree(d_Sx);
						cudaFree(d_Sy);
						cudaMalloc( (void**) &d_Sx, SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)) ;
						cudaMalloc( (void**) &d_Sy, SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)) ;
						/* zero signal buffer */
						cudaMemset(d_Sx, 0 ,SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)); /* only work for 0 */
						cudaMemset(d_Sy, 0 ,SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)); /* only work for 0 */
						/* set buffer */
						ippsFreeHF(Sxbuffer);
						ippsFreeHF(Sybuffer);
						Sxbuffer = ippsMalloc_32fHF(SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum));
						Sybuffer = ippsMalloc_32fHF(SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum));
					}

					/* avoid shared memory overflow */
					if (g_Sig.size() * sizeof(float) > deviceProp.sharedMemPerBlock){
						SBufferLen = 0;
					}else{
						SBufferLen = g_Sig.size() * sizeof(float);
					}

					/* upload GPU sequence */
					cudaMemcpy( d_Sig, 	&g_Sig[0], 	g_Sig.size() * sizeof(float),	cudaMemcpyHostToDevice ) ;

                    /* call GPU kernel for spin discrete precessing */

                    std::cout << "Grid: (" << dimGridImg.x << ", " << dimGridImg.y << ", " << dimGridImg.z << ")" << std::endl;
                    std::cout << "Block: (" << dimBlockImg.x << ", " << dimBlockImg.y << ", " << dimBlockImg.z << ")" << std::endl;
                    std::cout << "SBufferLen: " << SBufferLen << std::endl;

					BlochKernelNormalGPU<<< dimGridImg, dimBlockImg, SBufferLen >>>
										((float)*Gyro, d_CS, d_Rho, d_T1, d_T2, d_Mz, d_My, d_Mx,
										d_dB0, d_dWRnd, d_Gzgrid, d_Gygrid, d_Gxgrid, d_TxCoilmg, d_TxCoilpe, d_RxCoilx, d_RxCoily,
										d_Sig, (float)*RxCoilDefault, (float)*TxCoilDefault,
										d_Sx, d_Sy, (float)*rfRef, SignalLen, SBufferLen,
										SpinMxColNum, SpinMxRowNum, SpinMxSliceNum, *SpinNum, *TypeNum, *TxCoilNum, *RxCoilNum, g_Sig.size()/(5+3*(*TxCoilNum)));
                    
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                    std::cerr << "Launch error: " << cudaGetErrorString(err) << std::endl;
                    }
                                        
                    cudaError_t erer = cudaDeviceSynchronize();
                    if (erer != cudaSuccess) {
                    std::cerr << "Kernel launch failed: " << cudaGetErrorString(erer) << std::endl;
}
					g_Sig.clear();
					Signalptr = Signali; /* shift signal array pointer */
				}
				
				/* signal acquisition */
				if (SignalLen>0){
					/* get Sx, Sy buffer from GPU */
					cudaMemcpy( Sybuffer, d_Sy, SpinMxNum * SignalLen * (*RxCoilNum) * (*TypeNum) * sizeof(float), cudaMemcpyDeviceToHost ) ;
					cudaMemcpy( Sxbuffer, d_Sx, SpinMxNum * SignalLen * (*RxCoilNum) * (*TypeNum) * sizeof(float), cudaMemcpyDeviceToHost ) ;
					
					/* sum MR signal via openMP */
					for (Typei = 0; Typei < *TypeNum; Typei++){
						for (RxCoili = 0; RxCoili < *RxCoilNum; RxCoili++){  /* signal acquisition per Rx coil */
							#pragma omp parallel
							{   
								#pragma omp for private(j, s, p_Sx, p_Sy, buffer) 
								for (j=0; j < SignalLen; j++){
									
									if (j==0){
										*ActiveThreadNum = omp_get_num_threads();
									}
									
									s=Signali-SignalLen+j;
									p_Sx = Sx + (Typei*(*RxCoilNum)*(*SignalNum)+RxCoili*(*SignalNum)+s);
									p_Sy = Sy + (Typei*(*RxCoilNum)*(*SignalNum)+RxCoili*(*SignalNum)+s);
								
									ippsSum_32fHF(&Sxbuffer[Typei * (SpinMxNum * SignalLen * (*RxCoilNum)) + RxCoili * (SpinMxNum * SignalLen) +  j*SpinMxNum], SpinMxNum, &buffer);
									*p_Sx = (double)buffer;
									ippsSum_32fHF(&Sybuffer[Typei * (SpinMxNum * SignalLen * (*RxCoilNum)) + RxCoili * (SpinMxNum * SignalLen) +  j*SpinMxNum], SpinMxNum, &buffer);
									*p_Sy = (double)buffer;
								
								}
							}
						}       
					}
					
					/* zero signal buffer */
					cudaMemset(d_Sx, 0 ,SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)); /* only work for 0 */
					cudaMemset(d_Sy, 0 ,SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)); /* only work for 0 */
				}

				
				if (*gpuFetch !=0){
					/* fetch data from GPU */
					cudaMemcpy( Mz.data(), d_Mz, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( My.data(), d_My, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( Mx.data(), d_Mx, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( dWRnd, d_dWRnd, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( Rho.data(), d_Rho, SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( T1.data(), d_T1, SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( T2.data(), d_T2, SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( Gzgrid, d_Gzgrid, SpinMxNum * SpinMxSliceNum * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( Gygrid, d_Gygrid, SpinMxNum * SpinMxSliceNum * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( Gxgrid, d_Gxgrid, SpinMxNum * SpinMxSliceNum * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( dB0, d_dB0, SpinMxNum * SpinMxSliceNum * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( TxCoilmg, d_TxCoilmg, SpinMxNum * SpinMxSliceNum * (*TxCoilNum) * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( TxCoilpe, d_TxCoilpe, SpinMxNum * SpinMxSliceNum * (*TxCoilNum) * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( RxCoilx, d_RxCoilx, SpinMxNum * SpinMxSliceNum * (*RxCoilNum) * sizeof(float), cudaMemcpyDeviceToHost );
					cudaMemcpy( RxCoily, d_RxCoily, SpinMxNum * SpinMxSliceNum * (*RxCoilNum) * sizeof(float), cudaMemcpyDeviceToHost );
				}

                /* execute extended process */
                /*ExtCall = mexEvalString("DoExtPlugin");
                if (ExtCall){
                    mexErrMsgTxt("Extended process encounters ERROR!");
                    return;
                }*/
				
                /* update pointers, avoid pointer change between Matlab and Mex call */
                *t               += *dt;
                //*dt              = (double*) mxGetData(mxGetField(mexGetVariablePtr("global", "VVar"), 0, "dt"));
                *rfAmp           = rfAmpLine[i];
                *rfPhase         = rfPhaseLine[i];
                *rfFreq          = rfFreqLine[i];
                *rfCoil          = rfCoilLine[i];
                *rfRef           = 0; //rfRefLine[i];
                *GzAmp           = GzAmpLine[i];
                *GyAmp           = GyAmpLine[i];
                *GxAmp           = GxAmpLine[i];
                *ADC             = ADCLine[i];
                *Ext             = ExtLine[i];
                //*KzTmp           = (double*) mxGetData(mxGetField(mexGetVariablePtr("global", "VVar"), 0, "Kz"));
                //*KyTmp           = (double*) mxGetData(mxGetField(mexGetVariablePtr("global", "VVar"), 0, "Ky"));
                //*KxTmp           = (double*) mxGetData(mxGetField(mexGetVariablePtr("global", "VVar"), 0, "Kx"));
                //*gpuFetch     	= (double*) mxGetData(mxGetField(mexGetVariablePtr("global", "VVar"), 0, "gpuFetch"));
                *utsi            = i * 10;
                *rfi             = i;
                *Gzi             = i;
                *Gyi             = i;
                *Gxi             = i;
                *ADCi            = i;
                *Exti            = i;
                *TRCount         += 1;

				if (*gpuFetch !=0){
					*gpuFetch =0;
					/* update pointers, avoid pointer change between Matlab and Mex call */

                    /*Unchanging fields for now*/

					//Mz          = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VObj"), 0, "Mz"));
					//My          = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VObj"), 0, "My"));
					//Mx          = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VObj"), 0, "Mx"));
					//Rho         = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VObj"), 0, "Rho"));
					//T1          = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VObj"), 0, "T1"));
					//T2          = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VObj"), 0, "T2"));
				
                    //dWRnd       = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VMag"), 0, "dWRnd"));
					//dB0         = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VMag"), 0, "dB0"));
					//Gzgrid      = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VMag"), 0, "Gzgrid"));
					//Gygrid      = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VMag"), 0, "Gygrid"));
					//Gxgrid      = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VMag"), 0, "Gxgrid"));
					//TxCoilmg    = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VCoi"), 0, "TxCoilmg"));
					//TxCoilpe    = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VCoi"), 0, "TxCoilpe"));
					//RxCoilx     = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VCoi"), 0, "RxCoilx"));
					//RxCoily     = (float*) mxGetData(mxGetField(mexGetVariablePtr("global", "VCoi"), 0, "RxCoily"));

					/* send data back to GPU */
					cudaMemcpy( d_Mz, Mz.data(), SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_My, My.data(), SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_Mx, Mx.data(), SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_dWRnd, dWRnd, SpinMxNum * SpinMxSliceNum * (*SpinNum) * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_Rho, Rho.data(), SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_T1, T1.data(), SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_T2, T2.data(), SpinMxNum * SpinMxSliceNum * (*TypeNum) * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_Gzgrid, Gzgrid, SpinMxNum * SpinMxSliceNum * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_Gygrid, Gygrid, SpinMxNum * SpinMxSliceNum * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_Gxgrid, Gxgrid, SpinMxNum * SpinMxSliceNum * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_dB0, dB0, SpinMxNum * SpinMxSliceNum * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_TxCoilmg, TxCoilmg, SpinMxNum * SpinMxSliceNum * (*TxCoilNum) * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_TxCoilpe, TxCoilpe, SpinMxNum * SpinMxSliceNum * (*TxCoilNum) * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_RxCoilx, RxCoilx, SpinMxNum * SpinMxSliceNum * (*RxCoilNum) * sizeof(float), cudaMemcpyHostToDevice );
					cudaMemcpy( d_RxCoily, RxCoily, SpinMxNum * SpinMxSliceNum * (*RxCoilNum) * sizeof(float), cudaMemcpyHostToDevice );
				}
            }
            (*Exti)++;
        }

        //std::cout << "checkpoint4" << std::endl;
        
        if (flag[0]+flag[1]+flag[2]+flag[3]+flag[4]+flag[5] == 0){ /* reset VVar */
            ippsZero_64fHF(rfAmp, *TxCoilNum);
            ippsZero_64fHF(rfPhase, *TxCoilNum);
            ippsZero_64fHF(rfFreq, *TxCoilNum);
            *GzAmp = 0;
            *GyAmp = 0;
            *GxAmp = 0;
            *ADC = 0;
            *Ext = 0;
        }
        
		/* check TR point & end of time point */
		if (*dt <= 0){ 
			if (g_Sig.size() !=0){
				/* calculate signal length */
				SignalLen = Signali-Signalptr;

				/* reset buffer if needed */
				if (PreSignalLen!=SignalLen && SignalLen>0){
					PreSignalLen = SignalLen;
					/* allocate device memory for acquired signal buffer */
					cudaFree(d_Sx);
					cudaFree(d_Sy);
					cudaMalloc( (void**) &d_Sx, SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)) ;
					cudaMalloc( (void**) &d_Sy, SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)) ;
					/* zero signal buffer */
					cudaMemset(d_Sx, 0 ,SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)); /* only work for 0 */
					cudaMemset(d_Sy, 0 ,SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)); /* only work for 0 */
					/* set buffer */
					ippsFreeHF(Sxbuffer);
					ippsFreeHF(Sybuffer);
					Sxbuffer = ippsMalloc_32fHF(SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum));
					Sybuffer = ippsMalloc_32fHF(SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum));
				}

				/* avoid shared memory overflow */
				if (g_Sig.size() * sizeof(float) > deviceProp.sharedMemPerBlock){
					SBufferLen = 0;
				}else{
					SBufferLen = g_Sig.size() * sizeof(float);
				}

				/* upload GPU sequence */
				cudaMemcpy( d_Sig, 	&g_Sig[0], 	g_Sig.size() * sizeof(float),	cudaMemcpyHostToDevice ) ;

				/* call GPU kernel for spin discrete precessing */
				BlochKernelNormalGPU<<< dimGridImg, dimBlockImg, SBufferLen >>>
									((float)*Gyro, d_CS, d_Rho, d_T1, d_T2, d_Mz, d_My, d_Mx,
									d_dB0, d_dWRnd, d_Gzgrid, d_Gygrid, d_Gxgrid, d_TxCoilmg, d_TxCoilpe, d_RxCoilx, d_RxCoily,
									d_Sig, (float)*RxCoilDefault, (float)*TxCoilDefault,
									d_Sx, d_Sy, (float)*rfRef, SignalLen, SBufferLen,
									SpinMxColNum, SpinMxRowNum, SpinMxSliceNum, *SpinNum, *TypeNum, *TxCoilNum, *RxCoilNum, g_Sig.size()/(5+3*(*TxCoilNum)));
				cudaDeviceSynchronize();  
				g_Sig.clear();
				Signalptr = Signali;
			}
			
            
			/* signal acquisition */
			if (SignalLen>0){
				/* get Sx, Sy buffer from GPU */
				cudaMemcpy( Sybuffer, d_Sy, SpinMxNum * SignalLen * (*RxCoilNum) * (*TypeNum) * sizeof(float), cudaMemcpyDeviceToHost ) ;
				cudaMemcpy( Sxbuffer, d_Sx, SpinMxNum * SignalLen * (*RxCoilNum) * (*TypeNum) * sizeof(float), cudaMemcpyDeviceToHost ) ;
				
				/* sum MR signal via openMP */
				for (Typei = 0; Typei < *TypeNum; Typei++){
					for (RxCoili = 0; RxCoili < *RxCoilNum; RxCoili++){  /* signal acquisition per Rx coil */
						#pragma omp parallel
						{   
							#pragma omp for private(j, s, p_Sx, p_Sy, buffer) 
							for (j=0; j < SignalLen; j++){
								
								if (j==0){
									*ActiveThreadNum = omp_get_num_threads();
								}
								
								s=Signali-SignalLen+j;
								p_Sx = Sx + (Typei*(*RxCoilNum)*(*SignalNum)+RxCoili*(*SignalNum)+s);
								p_Sy = Sy + (Typei*(*RxCoilNum)*(*SignalNum)+RxCoili*(*SignalNum)+s);
							
								ippsSum_32fHF(&Sxbuffer[Typei * (SpinMxNum * SignalLen * (*RxCoilNum)) + RxCoili * (SpinMxNum * SignalLen) +  j*SpinMxNum], SpinMxNum, &buffer);
								*p_Sx = (double)buffer;
								ippsSum_32fHF(&Sybuffer[Typei * (SpinMxNum * SignalLen * (*RxCoilNum)) + RxCoili * (SpinMxNum * SignalLen) +  j*SpinMxNum], SpinMxNum, &buffer);
								*p_Sy = (double)buffer;
							
							}
						}
					}       
				}
				
				/* zero signal buffer */
				cudaMemset(d_Sx, 0 ,SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)); /* only work for 0 */
				cudaMemset(d_Sy, 0 ,SpinMxNum * SignalLen * (*TypeNum) * (*RxCoilNum) * sizeof(float)); /* only work for 0 */
			}

			if (*dt < 0){
				(*TRCount)++;
				/*mexPrintf("TR Counts: %d of %d\n", *TRCount, *TRNum);*/
			}
        }
        //std::cout << "checkpoint5" << std::endl;
    }

    /* free GPU memory */
    cudaFree(d_Mz);
    cudaFree(d_My);
    cudaFree(d_Mx);
    cudaFree(d_dWRnd);
    cudaFree(d_Rho);
    cudaFree(d_T1);
    cudaFree(d_T2);
    cudaFree(d_Gzgrid);
    cudaFree(d_Gygrid);
    cudaFree(d_Gxgrid);
    cudaFree(d_dB0);
    cudaFree(d_TxCoilmg);
    cudaFree(d_TxCoilpe);
	cudaFree(d_RxCoilx);
    cudaFree(d_RxCoily);
    cudaFree(d_CS);
    cudaFree(d_Sig);
	cudaFree(d_Sx);
	cudaFree(d_Sy);

    std::cout << "Mx beginning:" << std::endl;
    for (int i = 0; i < 100; i++) {
        if (Mx[i] != 0){
            std::cout << Mx[i] << " ";
        }
    }
    std::cout << "Mx middlle" << std::endl;
    for (int i = 300; i < 400; i++) {
            if (Mx[i] != 0){
                std::cout << Mx[i] << " ";
        }
    }
    std::cout << "Mx end:" << std::endl;
    for (int i = 900; i < 1000; i++) {
        if (Mx[i] != 0){
            std::cout << Mx[i] << " ";
        }
    }
    std::cout << "Simulation completed!" << std::endl;
	
	/* reset device, may slow down subsequent startup due to initialization */
	// cudaDeviceReset();
    return 0;
}    
