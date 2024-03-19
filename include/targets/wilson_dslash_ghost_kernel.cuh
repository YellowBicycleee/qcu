#pragma once

#include "basic_data/qcu_complex.cuh"
#include "basic_data/qcu_point.cuh"
#include "kernel/dslash_complex_product.cuh"
#include "kernel/public_kernels.cuh"

// COALESCED MEMORY
static __device__ __forceinline__ void loadGauge(Complex *u_local, void *gauge_ptr, int direction,
                                                 const Point &p, int sub_Lx, int Ly, int Lz,
                                                 int Lt) {
  Complex *u = p.getCoalescedGaugeAddr(gauge_ptr, direction, sub_Lx, Ly, Lz, Lt);
  int half_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[i] = *u;
    u += half_vol;
  }
  reconstructSU3(u_local);
}

// COALESCED
static __device__ __forceinline__ void loadVector(Complex *src_local, void *fermion_in,
                                                  const Point &p, int sub_Lx, int Ly, int Lz,
                                                  int Lt) {
  Complex *src = p.getCoalescedVectorAddr(fermion_in, sub_Lx, Ly, Lz, Lt);
  int half_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = *src;
    src += half_vol;
  }
}

// COALESCED:
static __device__ __forceinline__ void storeVector(Complex *dst_local, void *fermion_out,
                                                   const Point &p, int sub_Lx, int Ly, int Lz,
                                                   int Lt) {
  Complex *src = p.getCoalescedVectorAddr(fermion_out, sub_Lx, Ly, Lz, Lt);
  int half_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < Ns * Nc; i++) {
    *src = dst_local[i];
    src += half_vol;
  }
}

__global__ void DslashTransferFrontX(void *gauge, void *fermion_in, int Lx, int Ly, int Lz, int Lt,
                                     int parity, Complex *send_buffer, double dagger_flag) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;

  int new_even_odd = (z + t) & 0x01;
  Point p(sub_Lx - 1, 2 * sub_y + (new_even_odd == 1 - parity), z, t, 1 - parity);
  Point dst_p(0, sub_y, z, t, 0); // parity is useless
  // Complex* dst_ptr;

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, X_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  // even save to even, odd save to even
  // dst_ptr = send_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j].multipy_i() * dagger_flag) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp.multipy_minus_i() * dagger_flag;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[2 * Nc + j].multipy_i() * dagger_flag) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp.multipy_minus_i() * dagger_flag;
    }
  }

  // for (int i = 0; i < Ns * Nc; i++) {
  //   dst_ptr[i] = dst_local[i];
  // }
  // x轴与其他轴不同
  storeVector(dst_local, send_buffer, dst_p, 1, sub_Ly, Lz, Lt);
}

__global__ void DslashTransferBackX(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity,
                                    Complex *send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x; // 注意这里乘以2
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;

  int new_even_odd = (z + t) & 0x01;
  Point p(0, 2 * sub_y + (new_even_odd != 1 - parity), z, t, 1 - parity);
  Point dst_p(0, sub_y, z, t, 0); // parity is useless
  // Complex* dst_ptr;
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);

  // dst_ptr = send_buffer + thread * Ns * Nc;
  // for (int i = 0; i < Ns * Nc; i++) {
  //   dst_ptr[i] = src_local[i];
  // }
  storeVector(src_local, send_buffer, dst_p, 1, sub_Ly, Lz, Lt);
}
__global__ void DslashTransferFrontY(void *gauge, void *fermion_in, int Lx, int Ly, int Lz, int Lt,
                                     int parity, Complex *send_buffer, double dagger_flag) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, Ly - 1, z, t, 1 - parity);
  Point dst_p(x, 0, z, t, 0); // parity is useless

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Y_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j] * dagger_flag) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += -temp * dagger_flag;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[2 * Nc + j] * dagger_flag) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * dagger_flag;
    }
  }

  storeVector(dst_local, send_buffer, dst_p, sub_Lx, 1, Lz, Lt);
}
__global__ void DslashTransferBackY(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity,
                                    Complex *send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, 0, z, t, 1 - parity);
  Point dst_p(x, 0, z, t, 0); // parity is useless
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  storeVector(src_local, send_buffer, dst_p, sub_Lx, 1, Lz, Lt);
}
// DslashTransferFrontZ: DONE
__global__ void DslashTransferFrontZ(void *gauge, void *fermion_in, int Lx, int Ly, int Lz, int Lt,
                                     int parity, Complex *send_buffer, double dagger_flag) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, Lz - 1, t, 1 - parity);
  Point dst_p(x, y, 0, t, 0); // parity is useless

  Complex *dst_ptr;

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Z_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[2 * Nc + j].multipy_i() * dagger_flag) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp.multipy_minus_i() * dagger_flag;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j].multipy_i() * dagger_flag) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp.multipy_i() * dagger_flag;
    }
  }

  storeVector(dst_local, send_buffer, dst_p, sub_Lx, Ly, 1, Lt);
}
// DslashTransferBackZ: Done
__global__ void DslashTransferBackZ(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity,
                                    Complex *send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, 0, t, 1 - parity);
  Point dst_p(x, y, 0, t, 0);
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  storeVector(src_local, send_buffer, dst_p, sub_Lx, Ly, 1, Lt);
}

// DslashTransferFrontT: Done
__global__ void DslashTransferFrontT(void *gauge, void *fermion_in, int Lx, int Ly, int Lz, int Lt,
                                     int parity, Complex *send_buffer, double dagger_flag) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, Lt - 1, 1 - parity);
  Point dst_p(x, y, z, 0, 0); // parity is useless

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, T_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[2 * Nc + j] * dagger_flag) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * dagger_flag;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j] * dagger_flag) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * dagger_flag;
    }
  }

  storeVector(dst_local, send_buffer, dst_p, sub_Lx, Ly, Lz, 1);
}
// DslashTransferBackT: Done
__global__ void DslashTransferBackT(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity,
                                    Complex *send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, 0, 1 - parity);
  Point dst_p(x, y, z, 0, 0); // parity is useless

  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  storeVector(src_local, send_buffer, dst_p, sub_Lx, Ly, Lz, 1);
}

// ---separate line-----
// after this is postDslash kernels

__global__ void calculateBackBoundaryX(void *fermion_out, int Lx, int Ly, int Lz, int Lt,
                                       int parity, Complex *recv_buffer) {
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;

  int new_even_odd = (z + t) & 0x01; // %2
  Point p(0, 2 * sub_y + (new_even_odd != parity), z, t, parity);
  Point buffer_p(0, sub_y, z, t, 0); // parity is useless
  Complex *dst_ptr = p.getPointVector(static_cast<Complex *>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  loadVector(src_local, recv_buffer, buffer_p, 1, sub_Ly, Lz, Lt);

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  storeVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
}
__global__ void calculateFrontBoundaryX(void *gauge, void *fermion_out, int Lx, int Ly, int Lz,
                                        int Lt, int parity, Complex *recv_buffer,
                                        double dagger_flag) {
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x; // 注意这里乘以2
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;

  int new_even_odd = (z + t) & 0x01; // %2
  Point p(sub_Lx - 1, 2 * sub_y + (new_even_odd == parity), z, t, parity);
  Point buffer_p(0, sub_y, z, t, 0); // parity is useless
  Complex temp;
  Complex *dst_ptr = p.getPointVector(static_cast<Complex *>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, X_DIRECTION, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  loadVector(src_local, recv_buffer, buffer_p, 1, sub_Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j].multipy_i() * dagger_flag) *
             u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp.multipy_i() * dagger_flag;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j].multipy_i() * dagger_flag) *
             u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp.multipy_i() * dagger_flag;
    }
  }

  storeVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
}

__global__ void calculateBackBoundaryY(void *fermion_out, int Lx, int Ly, int Lz, int Lt,
                                       int parity, Complex *recv_buffer) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, 0, z, t, parity);
  Point buffer_p(x, 0, z, t, 0); // parity is useless
  Complex *dst_ptr = p.getPointVector(static_cast<Complex *>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  loadVector(src_local, recv_buffer, buffer_p, sub_Lx, 1, Lz, Lt);

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  storeVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
}

__global__ void calculateFrontBoundaryY(void *gauge, void *fermion_out, int Lx, int Ly, int Lz,
                                        int Lt, int parity, Complex *recv_buffer,
                                        double dagger_flag) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, Ly - 1, z, t, parity);
  Point buffer_p(x, 0, z, t, 0); // parity is useless
  Complex temp;
  Complex *dst_ptr = p.getPointVector(static_cast<Complex *>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, Y_DIRECTION, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  loadVector(src_local, recv_buffer, buffer_p, sub_Lx, 1, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j] * dagger_flag) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * dagger_flag;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j] * dagger_flag) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += -temp * dagger_flag;
    }
  }

  storeVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
}
__global__ void calculateBackBoundaryZ(void *fermion_out, int Lx, int Ly, int Lz, int Lt,
                                       int parity, Complex *recv_buffer) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, 0, t, parity);
  Point buffer_p(x, y, 0, t, 0); // parity is useless

  Complex *dst_ptr = p.getPointVector(static_cast<Complex *>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  loadVector(src_local, recv_buffer, buffer_p, sub_Lx, Ly, 1, Lt);

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  storeVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
}

__global__ void calculateFrontBoundaryZ(void *gauge, void *fermion_out, int Lx, int Ly, int Lz,
                                        int Lt, int parity, Complex *recv_buffer,
                                        double dagger_flag) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, Lz - 1, t, parity);
  Point buffer_p(x, y, 0, t, 0); // parity is useless
  Complex temp;
  Complex *dst_ptr = p.getPointVector(static_cast<Complex *>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, Z_DIRECTION, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  loadVector(src_local, recv_buffer, buffer_p, sub_Lx, Ly, 1, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j].multipy_i() * dagger_flag) *
             u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp.multipy_i() * dagger_flag;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j].multipy_i() * dagger_flag) *
             u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp.multipy_minus_i() * dagger_flag;
    }
  }

  storeVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
}

__global__ void calculateBackBoundaryT(void *fermion_out, int Lx, int Ly, int Lz, int Lt,
                                       int parity, Complex *recv_buffer) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, 0, parity);
  Point buffer_p(x, y, z, 0, 0); // parity is useless
  Complex *dst_ptr = p.getPointVector(static_cast<Complex *>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  loadVector(src_local, recv_buffer, buffer_p, sub_Lx, Ly, Lz, 1);

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  storeVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
}

__global__ void calculateFrontBoundaryT(void *gauge, void *fermion_out, int Lx, int Ly, int Lz,
                                        int Lt, int parity, Complex *recv_buffer,
                                        double dagger_flag) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, Lt - 1, parity);
  Point buffer_p(x, y, z, 0, 0); // parity is useless
  Complex temp;
  Complex *dst_ptr = p.getPointVector(static_cast<Complex *>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, 3, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  loadVector(src_local, recv_buffer, buffer_p, sub_Lx, Ly, Lz, 1);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j] * dagger_flag) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += -temp * dagger_flag;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j] * dagger_flag) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += -temp * dagger_flag;
    }
  }

  storeVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
}
