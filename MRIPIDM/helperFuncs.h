//ipps
#include <ipps.h>
#ifdef FW
#include <fwBase.h>
#include <fwSignal.h>

#define Ipp32f                  Fw32f                 
#define ippAlgHintFast          fwAlgHintFast
#define ippsMalloc_32f          fwsMalloc_32f
#define ippsFree                fwsFree
#define ippsZero_32f            fwsZero_32f
#define ippsZero_64f            fwsZero_64f
#define ippsSum_32f             fwsSum_32f
#define ippsCopy_32f            fwsCopy_32f
#define ippsAddC_32f            fwsAddC_32f
#define ippsAddC_32f_I          fwsAddC_32f_I
#define ippsAdd_32f             fwsAdd_32f 
#define ippsAdd_32f_I           fwsAdd_32f_I
#define ippsMulC_32f            fwsMulC_32f
#define ippsMulC_32f_I          fwsMulC_32f_I
#define ippsMul_32f             fwsMul_32f
#define ippsMul_32f_I           fwsMul_32f_I
#define ippsDiv_32f             fwsDiv_32f
#define ippsDivC_32f            fwsDivC_32f
#define ippsInv_32f_A24         fwsInv_32f_A24
#define ippsThreshold_LT_32f_I  fwsThreshold_LT_32f_I
#define ippsExp_32f_I           fwsExp_32f_I
#define ippsArctan_32f          fwsArctan_32f
#define ippsSqr_32f             fwsSqr_32f
#define ippsSqr_32f_I           fwsSqr_32f_I
#define ippsSqrt_32f_I          fwsSqrt_32f_I
#define ippsSin_32f_A24         fwsSin_32f_A24
#define ippsCos_32f_A24         fwsCos_32f_A24
#define ippsPolarToCart_32f     fwsPolarToCart_32f
#define ippsCartToPolar_32f     fwsCartToPolar_32f
#endif

#ifndef HELPERUNCS_H
#define HELPERUNCS_H

 float* ippsMalloc_32fHF(int len) ;
 void ippsFreeHF(float* ptr);
 void ippsZero_64fHF(float* ptr, int len);
 void ippsSum_32fHF(const float* src, int len, float* sum, IppAlgHint algHint = ippAlgHintFast);

#endif