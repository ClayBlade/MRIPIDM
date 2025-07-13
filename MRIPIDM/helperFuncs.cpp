#include BlochKernelGMGPU.h
#include <iostream>

extern "C" float* ippsMalloc_32fHF(int len) {
    return ippsMalloc_32f(len);
}

extern "C" void ippsFreeHF(float* ptr) {
    ippsFree(ptr);
}
extern "C" void ippsZero_64fHF(float* ptr, int len) {
    ippsZero_64f(ptr, len);
}
extern "C" void ippsSum_32fHF(const float* src, int len, float* sum, IppAlgHint algHint) {
    ippsSum_32f(src, len, sum);
}
