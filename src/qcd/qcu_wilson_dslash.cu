#include "qcd/qcu_wilson_dslash.cuh"
#include "targets/wilson_dslash_kernel.cuh"

BEGIN_NAMESPACE(qcu)

// use this function to call kernel function, this function donnot sync inside
// void WilsonDslash::apply(int dagge)) {
void WilsonDslash::apply(int daggerFlag) {
  void *gauge = dslashParam_->gauge;
  void *fermionIn = dslashParam_->fermionIn;
  void *fermionOut = dslashParam_->fermionOut;
  int parity = dslashParam_->parity;
  int Lx = dslashParam_->Lx;
  int Ly = dslashParam_->Ly;
  int Lz = dslashParam_->Lz;
  int Lt = dslashParam_->Lt;
  int Nx = dslashParam_->Nx;
  int Ny = dslashParam_->Ny;
  int Nz = dslashParam_->Nz;
  int Nt = dslashParam_->Nt;
  double daggerParam = daggerFlag ? -1.0 : 1.0;
  int vol = Lx * Ly * Lz * Lt;

  int gridSize = (vol / 2 + blockSize_ - 1) / blockSize_;

  dslashKernelFunc<<<gridSize, blockSize_, 0, dslashStream_>>>(
      gauge, fermionIn, fermionOut, Lx, Ly, Lz, Lt, parity, Nx, Ny, Nz, Nt, daggerParam);
  // static __global__ void dslashKernelFunc(void *gauge, void *fermion_in, void *fermion_out, int Lx,
  //                                      int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y,
  //                                      int grid_z, int grid_t, double flag) 
}

END_NAMESPACE(qcu)