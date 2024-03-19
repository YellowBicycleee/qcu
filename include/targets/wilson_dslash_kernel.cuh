#pragma once

#include "basic_data/qcu_complex.cuh"
#include "basic_data/qcu_point.cuh"
#include "qcu_macro.h"
#include "targets/dslash_complex_product.cuh"

static __global__ void dslashKernelFunc(void *gauge, void *fermion_in, void *fermion_out, int Lx,
                                        int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y,
                                        int grid_z, int grid_t, double flag) {
  assert(parity == 0 || parity == 1);
  Lx >>= 1;

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
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
  loadGaugeCoalesced(u_local, gauge, X_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, X_DIRECTION, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  // x front    x == Lx-1 && parity != eo
  coord_boundary = (grid_x > 1 && x == Lx - 1 && parity != eo) ? Lx - 1 : Lx;
  if (x < coord_boundary) {
    spinor_gauge_mul_add_vec<X_DIRECTION, FRONT>(u_local, src_local, dst_local, flag);
  }

  // x back   x==0 && parity == eo
  move_point = p.move(BACK, X_DIRECTION, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, X_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_x > 1 && x == 0 && parity == eo) ? 1 : 0;
  if (x >= coord_boundary) {
    spinor_gauge_mul_add_vec<X_DIRECTION, BACK>(u_local, src_local, dst_local, flag);
  }

  // \mu = 2
  // y front
  loadGaugeCoalesced(u_local, gauge, Y_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, Y_DIRECTION, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_y > 1) ? Ly - 1 : Ly;
  if (y < coord_boundary) {
    spinor_gauge_mul_add_vec<Y_DIRECTION, FRONT>(u_local, src_local, dst_local, flag);
  }

  // y back
  move_point = p.move(BACK, Y_DIRECTION, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, Y_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_y > 1) ? 1 : 0;
  if (y >= coord_boundary) {
    spinor_gauge_mul_add_vec<Y_DIRECTION, BACK>(u_local, src_local, dst_local, flag);
  }

  // \mu = 3
  // z front
  loadGaugeCoalesced(u_local, gauge, Z_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, Z_DIRECTION, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_z > 1) ? Lz - 1 : Lz;
  if (z < coord_boundary) {
    spinor_gauge_mul_add_vec<Z_DIRECTION, FRONT>(u_local, src_local, dst_local, flag);
  }

  // z back
  move_point = p.move(BACK, Z_DIRECTION, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, Z_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_z > 1) ? 1 : 0;
  if (z >= coord_boundary) {
    spinor_gauge_mul_add_vec<Z_DIRECTION, BACK>(u_local, src_local, dst_local, flag);
  }

  // t: front
  // loadGauge(u_local, gauge, 3, p, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, T_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, T_DIRECTION, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? Lt - 1 : Lt;
  if (t < coord_boundary) {
    spinor_gauge_mul_add_vec<T_DIRECTION, FRONT>(u_local, src_local, dst_local, flag);
  }

  // t: back
  move_point = p.move(BACK, 3, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, T_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? 1 : 0;
  if (t >= coord_boundary) {
    spinor_gauge_mul_add_vec<T_DIRECTION, BACK>(u_local, src_local, dst_local, flag);
  }

  // store result
  storeVectorCoalesced(dst_local, fermion_out, p, Lx, Ly, Lz, Lt);
}