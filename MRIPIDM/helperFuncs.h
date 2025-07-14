#ifdef __CUDACC__  // If compiling with nvcc (CUDA)
    typedef int IppAlgHint;
    #define ippAlgHintFast 0
#else
    #include <ipps.h>
#endif

#ifndef HELPERFUNCS_H
#define HELPERFUNCS_H

 float* ippsMalloc_32fHF(int len) ;
 void ippsFreeHF(float* ptr);
 void ippsZero_64fHF(float* ptr, int len);
 void ippsSum_32fHF(const float* src, int len, float* sum, ippAlgHintFast);

#endif