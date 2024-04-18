#pragma once

#include "basic_data/qcu_complex.cuh"
#include "basic_data/qcu_point.cuh"
#include "qcu_macro.cuh"
#include "targets/dslash_complex_product.cuh"

static __global__ void dslashKernelFunc(void *gauge, void *fermion_in, void *fermion_out, int Lx,
                                        int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y,
                                        int grid_z, int grid_t, double flag) {
  assert(parity == 0 || parity == 1);
  Lx >>= 1;

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  // if (thread_id == 0) {
  //   printf("grid[4] = {%d, %d, %d, %d}\n", grid_x, grid_y, grid_z, grid_t);
  // }
  int t = thread_id / (Lz * Ly * Lx);
  int z = thread_id % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread_id % (Ly * Lx) / Lx;
  int x = thread_id % Lx;

  int coord_boundary;

  Point p(x, y, z, t, parity);
  Point move_point;
  Complex u_local[Nc * Nc];   // for GPU
  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU

  int eo = (y + z + t) & 0x01;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  // \mu = 1
  // loadGauge(u_local, gauge, 0, p, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, X_DIM, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FWD, X_DIM, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  // x front    x == Lx-1 && parity != eo
  coord_boundary = (grid_x > 1 && x == Lx - 1 && parity != eo) ? Lx - 1 : Lx;
  if (x < coord_boundary) {
    spinor_gauge_mul_add_vec<X_DIM, FWD>(u_local, src_local, dst_local, flag);
  }

  // x back   x==0 && parity == eo
  move_point = p.move(BWD, X_DIM, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, X_DIM, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_x > 1 && x == 0 && parity == eo) ? 1 : 0;
  if (x >= coord_boundary) {
    spinor_gauge_mul_add_vec<X_DIM, BWD>(u_local, src_local, dst_local, flag);
  }

  // \mu = 2
  // y front
  loadGauge(u_local, gauge, Y_DIM, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FWD, Y_DIM, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_y > 1) ? Ly - 1 : Ly;
  if (y < coord_boundary) {
    spinor_gauge_mul_add_vec<Y_DIM, FWD>(u_local, src_local, dst_local, flag);
  }

  // y back
  move_point = p.move(BWD, Y_DIM, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Y_DIM, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_y > 1) ? 1 : 0;
  if (y >= coord_boundary) {
    spinor_gauge_mul_add_vec<Y_DIM, BWD>(u_local, src_local, dst_local, flag);
  }

  // \mu = 3
  // z front
  loadGauge(u_local, gauge, Z_DIM, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FWD, Z_DIM, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_z > 1) ? Lz - 1 : Lz;
  if (z < coord_boundary) {
    spinor_gauge_mul_add_vec<Z_DIM, FWD>(u_local, src_local, dst_local, flag);
  }

  // z back
  move_point = p.move(BWD, Z_DIM, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Z_DIM, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_z > 1) ? 1 : 0;
  if (z >= coord_boundary) {
    spinor_gauge_mul_add_vec<Z_DIM, BWD>(u_local, src_local, dst_local, flag);
  }

  // t: front
  // loadGauge(u_local, gauge, 3, p, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, T_DIM, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FWD, T_DIM, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? Lt - 1 : Lt;
  if (t < coord_boundary) {
    spinor_gauge_mul_add_vec<T_DIM, FWD>(u_local, src_local, dst_local, flag);
  }

  // t: back
  move_point = p.move(BWD, T_DIM, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, T_DIM, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? 1 : 0;
  if (t >= coord_boundary) {
    spinor_gauge_mul_add_vec<T_DIM, BWD>(u_local, src_local, dst_local, flag);
  }

  // store result
  storeVector(dst_local, fermion_out, p, Lx, Ly, Lz, Lt);
}

// template <int _dir, int _fb, QCU_DAGGER_FLAG daggerFlag>  // _dir from 0-3 means X-Z
// static __device__ __forceinline__ void dslashMVKernel(Complex *u_local, Complex *src_local, Complex *dst_local)

// template <int _DaggerFlag>
// static __global__ void dslashKernelFunc(void *gauge, void *fermion_in, void *fermion_out, int Lx, int Ly, int Lz,
//                                         int Lt, int parity, int grid_x, int grid_y, int grid_z, int grid_t) {
//   assert(parity == 0 || parity == 1);
//   Lx >>= 1;
//   int daggerFlag;
//   if (_DaggerFlag == 0)
//     daggerFlag = 1;
//   else
//     daggerFlag = -1;

//   int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//   int t = thread_id / (Lz * Ly * Lx);
//   int z = thread_id % (Lz * Ly * Lx) / (Ly * Lx);
//   int y = thread_id % (Ly * Lx) / Lx;
//   int x = thread_id % Lx;

//   int coord_boundary;

//   Point p(x, y, z, t, parity);
//   Point move_point;
//   Complex u_local[Nc * Nc];    // for GPU
//   Complex src_local[Ns * Nc];  // for GPU
//   Complex dst_local[Ns * Nc];  // for GPU

//   int eo = (y + z + t) & 0x01;

//   for (int i = 0; i < Ns * Nc; i++) {
//     dst_local[i].clear2Zero();
//   }

//   // \mu = 1
//   loadGauge(u_local, gauge, X_DIM, p, Lx, Ly, Lz, Lt);
//   move_point = p.move(FWD, X_DIM, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
//   // x front    x == Lx-1 && parity != eo
//   coord_boundary = (grid_x > 1 && x == Lx - 1 && parity != eo) ? Lx - 1 : Lx;
//   if (x < coord_boundary) {
//     dslashMVKernel<X_DIM, FWD, _DaggerFlag>(u_local, src_local, dst_local);
//     // spinor_gauge_mul_add_vec<X_DIM, FWD>(u_local, src_local, dst_local, daggerFlag);
//   }

//   // x back   x==0 && parity == eo
//   move_point = p.move(BWD, X_DIM, Lx, Ly, Lz, Lt);
//   loadGauge(u_local, gauge, X_DIM, move_point, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   coord_boundary = (grid_x > 1 && x == 0 && parity == eo) ? 1 : 0;
//   if (x >= coord_boundary) {
//     dslashMVKernel<X_DIM, BWD, _DaggerFlag>(u_local, src_local, dst_local);
//     // spinor_gauge_mul_add_vec<X_DIM, BWD>(u_local, src_local, dst_local, daggerFlag);
//   }

//   // \mu = 2
//   // y front
//   loadGauge(u_local, gauge, Y_DIM, p, Lx, Ly, Lz, Lt);
//   move_point = p.move(FWD, Y_DIM, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   coord_boundary = (grid_y > 1) ? Ly - 1 : Ly;
//   if (y < coord_boundary) {
//     dslashMVKernel<Y_DIM, FWD, _DaggerFlag>(u_local, src_local, dst_local);
//     // spinor_gauge_mul_add_vec<Y_DIM, FWD>(u_local, src_local, dst_local, daggerFlag);
//   }

//   // y back
//   move_point = p.move(BWD, Y_DIM, Lx, Ly, Lz, Lt);
//   loadGauge(u_local, gauge, Y_DIM, move_point, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   coord_boundary = (grid_y > 1) ? 1 : 0;
//   if (y >= coord_boundary) {
//     dslashMVKernel<Y_DIM, BWD, _DaggerFlag>(u_local, src_local, dst_local);
//     // spinor_gauge_mul_add_vec<Y_DIM, BWD>(u_local, src_local, dst_local, daggerFlag);
//   }

//   // \mu = 3
//   // z front
//   loadGauge(u_local, gauge, Z_DIM, p, Lx, Ly, Lz, Lt);
//   move_point = p.move(FWD, Z_DIM, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
//   coord_boundary = (grid_z > 1) ? Lz - 1 : Lz;
//   if (z < coord_boundary) {
//     dslashMVKernel<Z_DIM, FWD, _DaggerFlag>(u_local, src_local, dst_local);
//     // spinor_gauge_mul_add_vec<Z_DIM, FWD>(u_local, src_local, dst_local, daggerFlag);
//   }

//   // z back
//   move_point = p.move(BWD, Z_DIM, Lx, Ly, Lz, Lt);
//   loadGauge(u_local, gauge, Z_DIM, move_point, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   coord_boundary = (grid_z > 1) ? 1 : 0;
//   if (z >= coord_boundary) {
//     dslashMVKernel<Z_DIM, BWD, _DaggerFlag>(u_local, src_local, dst_local);
//     // spinor_gauge_mul_add_vec<Z_DIM, BWD>(u_local, src_local, dst_local, daggerFlag);
//   }

//   // t: front
//   // loadGauge(u_local, gauge, 3, p, Lx, Ly, Lz, Lt);
//   loadGauge(u_local, gauge, T_DIM, p, Lx, Ly, Lz, Lt);
//   move_point = p.move(FWD, T_DIM, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   coord_boundary = (grid_t > 1) ? Lt - 1 : Lt;
//   if (t < coord_boundary) {
//     dslashMVKernel<T_DIM, FWD, _DaggerFlag>(u_local, src_local, dst_local);
//   }

//   // t: back
//   move_point = p.move(BWD, T_DIM, Lx, Ly, Lz, Lt);
//   loadGauge(u_local, gauge, T_DIM, move_point, Lx, Ly, Lz, Lt);
//   loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

//   coord_boundary = (grid_t > 1) ? 1 : 0;
//   if (t >= coord_boundary) {
//     dslashMVKernel<T_DIM, BWD, _DaggerFlag>(u_local, src_local, dst_local);
//   }

//   // store result
//   storeVector(dst_local, fermion_out, p, Lx, Ly, Lz, Lt);
// }