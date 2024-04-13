#pragma once

#include "algebra/qcu_algebra.h"
#include "comm/qcu_communicator.h"
#include "mempool/qcu_mempool.h"
#include "qcu.h"
#include "qcu_macro.cuh"
#include <cuda.h>
BEGIN_NAMESPACE(qcu)

enum DslashType {
  WILSON_DSLASH_4D = 0,
  CLOVER_DSLASH_4D,

};

struct QcuParam {};

struct DslashParam : public QcuParam {
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

  const double kappa;
  void *fermionIn;
  void *fermionOut;
  void *gauge;
  void *tempFermionIn1; // use this ptr to store the temp fermion field, when calc dagger and
                        // non-dagger dslash, single dslash assign this variable to nullptr,
                        // when using cg solver, cg solver will assign this variable to a temp space
  void *tempFermionIn2;

  QcuMemPool *memPool;
  MsgHandler *msgHandler;
  QcuComm *qcuComm;

  cudaStream_t stream1;
  cudaStream_t stream2;

  // constructor
  DslashParam(void *pFermionIn, void *pFermionOut, void *pGauge, int pLx, int pLy, int pLz, int pLt, int pParity,
              int pNx, int pNy, int pNz, int pNt, double pKappa, int pDaggerFlag = 0, QcuMemPool *pMemPool = nullptr,
              MsgHandler *pMsgHandler = nullptr, QcuComm *pQcuComm = nullptr, cudaStream_t pStream1 = NULL,
              cudaStream_t pStream2 = NULL)
      : fermionIn(pFermionIn), fermionOut(pFermionOut), gauge(pGauge), Lx(pLx), Ly(pLy), Lz(pLz), Lt(pLt),
        parity(pParity), Nx(pNx), Ny(pNy), Nz(pNz), Nt(pNt), kappa(pKappa), daggerFlag(pDaggerFlag), memPool(pMemPool),
        msgHandler(pMsgHandler), qcuComm(pQcuComm), tempFermionIn1(nullptr), tempFermionIn2(nullptr), stream1(pStream1),
        stream2(pStream2) {
#ifdef DEBUG
    printf("In DslashParam constructor, pKappa = %lf, kappa = %lf\n", pKappa, kappa);
#endif
  }

  // copy constructor
  DslashParam(const DslashParam &rhs)
      : fermionIn(rhs.fermionIn), fermionOut(rhs.fermionOut), gauge(rhs.gauge), Lx(rhs.Lx), Ly(rhs.Ly), Lz(rhs.Lz),
        Lt(rhs.Lt), parity(rhs.parity), Nx(rhs.Nx), Ny(rhs.Ny), Nz(rhs.Nz), Nt(rhs.Nt), kappa(rhs.kappa),
        daggerFlag(rhs.daggerFlag), memPool(rhs.memPool), msgHandler(rhs.msgHandler), qcuComm(rhs.qcuComm),
        tempFermionIn1(rhs.tempFermionIn1), tempFermionIn2(rhs.tempFermionIn2), stream1(rhs.stream1),
        stream2(rhs.stream2) {}

  void changeParity() { parity = 1 - parity; }
};

// host class, to call kernel functions
class Dslash {
protected:
  int blockSize_;

public:
  DslashParam *dslashParam_;
  Dslash(DslashParam *param, int blockSize = 256) : dslashParam_(param), blockSize_(blockSize) {}
  virtual ~Dslash() {}
  virtual void apply() = 0;
  virtual void preApply() = 0;
  virtual void postApply() = 0;
};

struct DslashMV : QcuSPMV {

  Dslash *dslash;
  DslashMV(Dslash *pDslash = nullptr, int blockSize = 256) : dslash(pDslash), QcuSPMV(blockSize) {}
  DslashMV(const DslashMV &rhs) : dslash(rhs.dslash), QcuSPMV(rhs.blockSize) {}
  void operator=(const DslashMV &rhs) {
    dslash = rhs.dslash;
    blockSize = rhs.blockSize;
  }
  // 去掉了even_odd参数，
  // 此函数只负责传入result和src指针，其他参数，由dslashParam_传入
  // 待删除参数：stream
  virtual void operator()(_genvector result, _genvector src, cudaStream_t stream = NULL);
};

END_NAMESPACE(qcu)