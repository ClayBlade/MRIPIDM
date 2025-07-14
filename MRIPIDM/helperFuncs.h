#ifndef helperFuncs_H
#define helperFuncs_H

extern 'c' {
 float* ippsMalloc_32fHF(int len) ;
 void ippsFreeHF(float* ptr);
 void ippsZero_64fHF(double* ptr, int len);
 void ippsSum_32fHF(const float* src, int len, float* sum);
}
#endif