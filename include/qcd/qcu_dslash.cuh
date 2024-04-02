#pragma once

#include "comm/qcu_communicator.h"
#include "mempool/qcu_mempool.h"
#include "qcu.h"
#include "qcu_macro.cuh"
#include <cuda.h>
#include "algebra/qcu_algebra.h"
BEGIN_NAMESPACE(qcu)

enum DslashType {
  WILSON_DSLASH_4D = 0,
  CLOVER_DSLASH_4D,

};
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
  int daggerFlag;

  void *fermionIn;
  void *fermionOut;
  void *gauge;

  QcuMemPool *memPool;
  MsgHandler *msgHandler;
  QcuComm *qcuComm;
  // constructor
  DslashParam(void *pFermionIn, void *pFermionOut, void *pGauge, int pLx, int pLy, int pLz, int pLt,
              int pParity, int pNx, int pNy, int pNz, int pNt, int pDaggerFlag = 0,
              QcuMemPool *pMemPool = nullptr, MsgHandler *pMsgHandler = nullptr, QcuComm *pQcuComm = nullptr)
      : fermionIn(pFermionIn), fermionOut(pFermionOut), gauge(pGauge), Lx(pLx), Ly(pLy), Lz(pLz),
        Lt(pLt), parity(pParity), Nx(pNx), Ny(pNy), Nz(pNz), Nt(pNt),
        daggerFlag(pDaggerFlag), memPool(pMemPool), msgHandler(pMsgHandler), qcuComm(pQcuComm) {}

  // copy constructor
  DslashParam(const DslashParam &rhs)
      : fermionIn(rhs.fermionIn), fermionOut(rhs.fermionOut), gauge(rhs.gauge), Lx(rhs.Lx),
        Ly(rhs.Ly), Lz(rhs.Lz), Lt(rhs.Lt), parity(rhs.parity), Nx(rhs.Nx), Ny(rhs.Ny), Nz(rhs.Nz),
        Nt(rhs.Nt), daggerFlag(rhs.daggerFlag), memPool(rhs.memPool), msgHandler(rhs.msgHandler),
        qcuComm(rhs.qcuComm) {}

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
  void changeParity() {
    parity = 1 - parity;
  }
};

// host class, to call kernel functions
class Dslash {
protected:
  int blockSize_;

  DslashParam *dslashParam_;
  cudaStream_t cudaStream1_; // stream for dslash kernel, default stream is NULL
  cudaStream_t cudaStream2_; // stream for dslash kernel, default stream is NULL
public:
  Dslash(DslashParam *param, int blockSize = 256, cudaStream_t dslashStream1 = NULL,
         cudaStream_t dslashStream2 = NULL)
      : dslashParam_(param), blockSize_(blockSize), cudaStream1_(dslashStream1),
        cudaStream2_(dslashStream2) {}

  // use this function to call kernel function, this function donnot sync inside
  // virtual void apply(int daggerFlag) = 0;
  // virtual void preApply(int daggerFlag) = 0;
  // virtual void postApply(int daggerFlag) = 0;
  virtual void apply() = 0;
  virtual void preApply() = 0;
  virtual void postApply() = 0;
};


struct DslashMV : QcuSPMV {
  // const DslashType dslashType_;
 

  // DslashMV(const Dslash *dslash) : dslash(pDslash) {}
  virtual void operator()(_genvector fermionOut, _genvector gauge, _genvector fermionIn,
                          int parity = 0, cudaStream_t stream = NULL) {
    // int daggerFlag = dslash->dslashParam_->daggerFlag;
    // dslash->preApply();
    // dslash->apply();
    // dslash->postApply();
  }
};

END_NAMESPACE(qcu)