#pragma once
#include "qcu_complex.cuh"

static __device__ __forceinline__ void reconstructSU3(Complex *su3) {
  su3[6] = (su3[1] * su3[5] - su3[2] * su3[4]).conj();
  su3[7] = (su3[2] * su3[3] - su3[0] * su3[5]).conj();
  su3[8] = (su3[0] * su3[4] - su3[1] * su3[3]).conj();
}

// only use this function when dst and src are both register variables
static __device__ __forceinline__ void copyGauge(Complex *dst, Complex *src) {
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    dst[i] = src[i];
  }
  reconstructSU3(dst);
}