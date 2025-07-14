

#ifndef HELPERFUNCS_H
#define HELPERFUNCS_H
#include <ipps.h>
// Dummy placeholder to avoid compilation error

 float* ippsMalloc_32fHF(int len) ;
 void ippsFreeHF(float* ptr);
 void ippsZero_64fHF(float* ptr, int len);
 void ippsSum_32fHF(const float* src, int len, float* sum, ippAlgHintFast);

#endif

#ifdef __CUDACC__  // If compiling with nvcc (CUDA)
    typedef int IppAlgHint;
    #define ippAlgHintFast 0
#endif