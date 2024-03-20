#pragma once

#include "qcu.h"
#include "qcu_macro.cuh"
#include <cuda.h>

BEGIN_NAMESPACE(qcu)

struct DslashParam {
  int Lx;
  int Ly;
  int Lz;
  int Lt;
  int parity;

  int Nx;
  int Ny;
  int Nz;
  int Nt;

  void *fermionIn;
  void *fermionOut;
  void *gauge;

  // constructor
  DslashParam(void *pFermionIn, void *pFermionOut, void *pGauge, int pLx, int pLy, int pLz, int pLt,
              int p_parity, int pNx, int pNy, int pNz, int pNt)
      : fermionIn(pFermionIn), fermionOut(pFermionOut), gauge(pGauge), Lx(pLx), Ly(pLy), Lz(pLz),
        Lt(pLt), parity(p_parity), Nx(pNx), Ny(pNy), Nz(pNz), Nt(pNt) {}

  // copy constructor
  DslashParam(const DslashParam &rhs)
      : fermionIn(rhs.fermionIn), fermionOut(rhs.fermionOut), gauge(rhs.gauge), Lx(rhs.Lx),
        Ly(rhs.Ly), Lz(rhs.Lz), Lt(rhs.Lt), parity(rhs.parity) {}

  // copy assignment
  DslashParam &operator=(const DslashParam &rhs) {
    fermionIn = rhs.fermionIn;
    fermionOut = rhs.fermionOut;
    gauge = rhs.gauge;
    Lx = rhs.Lx;
    Ly = rhs.Ly;
    Lz = rhs.Lz;
    Lt = rhs.Lt;
    parity = rhs.parity;
    return *this;
  }
};

// host class, to call kernel functions
class Dslash {
protected:
  int blockSize_;

  DslashParam *dslashParam_;
  cudaStream_t dslashStream_; // stream for dslash kernel, default stream is NULL

public:
  Dslash(DslashParam* param, int blockSize = 256, cudaStream_t dslashStream = NULL)
      : dslashParam_(param), blockSize_(blockSize), dslashStream_(dslashStream) {}

  // use this function to call kernel function, this function donnot sync inside
  virtual void apply(int daggerFlag) = 0;
};

END_NAMESPACE(qcu)