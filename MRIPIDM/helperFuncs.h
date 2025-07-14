

#ifndef HELPERUNCS_H
#define HELPERUNCS_H
// Dummy placeholder to avoid compilation error

 float* ippsMalloc_32fHF(int len) ;
 void ippsFreeHF(float* ptr);
 void ippsZero_64fHF(float* ptr, int len);
 void ippsSum_32fHF(const float* src, int len, float* sum, ippAlgHintFast);

#endif