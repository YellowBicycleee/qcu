#pragma once
#include "basic_data/qcu_complex.cuh"

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

// COALESCED MEMORY
static __device__ __forceinline__ void loadGauge(Complex *u_local, void *gauge_ptr, int direction, const Point &p,
                                                 int sub_Lx, int Ly, int Lz, int Lt) {
  Complex *u = p.getCoalescedGaugeAddr(gauge_ptr, direction, sub_Lx, Ly, Lz, Lt);
  int half_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[i] = *u;
    u += half_vol;
  }
  reconstructSU3(u_local);
}
// version 2 does not reconstruct the SU3 matrix, only stores the 12 complex numbers
static __device__ __forceinline__ void loadGauge2(Complex *u_local, void *gauge_ptr, int direction, const Point &p,
                                                  int sub_Lx, int Ly, int Lz, int Lt) {
  Complex *u = p.getCoalescedGaugeAddr(gauge_ptr, direction, sub_Lx, Ly, Lz, Lt);
  int half_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[i] = *u;
    u += half_vol;
  }
}
static __device__ __forceinline__ void storeGauge(void *gauge_ptr, Complex *u_local, int direction, const Point &p,
                                                  int sub_Lx, int Ly, int Lz, int Lt) {
  Complex *u = p.getCoalescedGaugeAddr(gauge_ptr, direction, sub_Lx, Ly, Lz, Lt);
  int half_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    *u = u_local[i];
    u += half_vol;
  }
}

// COALESCED
static __device__ __forceinline__ void loadVector(Complex *src_local, void *fermion_in, const Point &p, int sub_Lx,
                                                  int Ly, int Lz, int Lt) {
  Complex *src = p.getCoalescedVectorAddr(fermion_in, sub_Lx, Ly, Lz, Lt);
  int half_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = *src;
    src += half_vol;
  }
}

// COALESCED:
static __device__ __forceinline__ void storeVector(Complex *dst_local, void *fermion_out, const Point &p, int sub_Lx,
                                                   int Ly, int Lz, int Lt) {
  Complex *src = p.getCoalescedVectorAddr(fermion_out, sub_Lx, Ly, Lz, Lt);
  int half_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < Ns * Nc; i++) {
    *src = dst_local[i];
    src += half_vol;
  }
}