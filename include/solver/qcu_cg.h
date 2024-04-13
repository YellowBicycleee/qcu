#pragma once
#include "comm/qcu_communicator.h"
#include "mempool/qcu_mempool.h"
#include "qcd/qcu_dslash.cuh"
#include "qcd/qcu_wilson_dslash.cuh"
#include "qcu_macro.cuh"
#include <assert.h>

BEGIN_NAMESPACE(qcu)

// use this to calc
struct CGDslashMV_Odd : QcuSPMV {

  Dslash *dslash;
  CGDslashMV_Odd() : dslash(nullptr), QcuSPMV(256) {}
  CGDslashMV_Odd(Dslash *pDslash, int blockSize = 256) : dslash(pDslash), QcuSPMV(blockSize) {}
  CGDslashMV_Odd(const CGDslashMV_Odd &cgDslashMV_Odd)
      : dslash(cgDslashMV_Odd.dslash), QcuSPMV(cgDslashMV_Odd.blockSize) {}
  // 去掉了even_odd参数，qcu里默认使用dslash的dslashParam_里的parity，且不改变
  virtual void operator()(_genvector result, _genvector src, cudaStream_t stream = NULL);
};

struct CGParam : public QcuParam {
  void *fermionInB;  // full fermion : even + odd
  void *fermionOutX; // full fermion : even + odd
  void *gauge;
  void *cloverMatrix;       // clover term
  void *cloverInvertMatrix; // clover term inverse

  //   double mass;
  double kappa;

  int Lx;
  int Ly;
  int Lz;
  int Lt;
  //   int parity;

  int Nx;
  int Ny;
  int Nz;
  int Nt;

  cudaStream_t stream1;
  cudaStream_t stream2;

  QcuMemPool *memPool;
  MsgHandler *msgHandler;
  QcuComm *qcuComm;

  CGParam(void *pFermionInB, void *pFermionOutX, void *pFermionGauge, void *pCloverMatrix, void *pCloverInvertMatrix,
          double pKappa, int pLx, int pLy, int pLz, int pLt, int pNx, int pNy, int pNz, int pNt, QcuMemPool *pMemPool,
          MsgHandler *pMsgHandler, QcuComm *pQcuComm, cudaStream_t pStream1 = NULL, cudaStream_t pStream2 = NULL)
      : fermionInB(pFermionInB), fermionOutX(pFermionOutX), gauge(pFermionGauge), cloverMatrix(pCloverMatrix),
        cloverInvertMatrix(pCloverInvertMatrix), kappa(pKappa), Lx(pLx), Ly(pLy), Lz(pLz), Lt(pLt), Nx(pNx), Ny(pNy),
        Nz(pNz), Nt(pNt), memPool(pMemPool), msgHandler(pMsgHandler), qcuComm(pQcuComm), stream1(pStream1),
        stream2(pStream2) {}

  CGParam(DslashParam *dslashParam, double pMass)
      : fermionInB(dslashParam->fermionIn), fermionOutX(dslashParam->fermionOut), gauge(dslashParam->gauge),
        cloverMatrix(nullptr), cloverInvertMatrix(nullptr), kappa(dslashParam->kappa), Lx(dslashParam->Lx),
        Ly(dslashParam->Ly), Lz(dslashParam->Lz), Lt(dslashParam->Lt), Nx(dslashParam->Nx), Ny(dslashParam->Ny),
        Nz(dslashParam->Nz), Nt(dslashParam->Nt), memPool(dslashParam->memPool), msgHandler(dslashParam->msgHandler),
        qcuComm(dslashParam->qcuComm) {}
};

class QcuCG {
  int blockSize_;
  int numIterations_; // 实际迭代次数_
  int maxIterations_; // 最大迭代次数

  double rsdTarget_; // 要求的相对误_差
  DSLASH_TYPE dslashType_;
  Dslash *dslash_;
  CGParam *cgParam_;
  DslashParam *dslashParam_;
  // FUNCTORS;
  QcuVectorAdd vectorAdd_;
  QcuInnerProd innerProd_; // reduce
  QcuNorm2 norm2_;         // reduce
  QcuSaxpy saxpy_;
  QcuSax mySax_;
  QcuComplexCopy complexCopy_;
  DslashMV singleDslash_;       // TODO: initialize
  CGDslashMV_Odd cgIterMV_Odd_; // odd A

  // void *evenFermionIn_;
  // void *oddFermionIn_;
  void *evenB_;
  void *oddB_;
  void *evenX_;
  void *oddX_;
  // temp vectors
  void *newEvenB_;
  void *newOddB_;
  void *residual_;
  void *pVec_;
  void *tmp1_;
  void *tmp2_;
  void *tmp3_;
  void *tmp4_;
  void *tmp5_;
  // for dslash temp_vector
  void *dslashTempVec1_;
  void *dslashTempVec2_;

  void allocateTempVectors();
  void freeTempVectors();

  bool odd_cg(void *resX, void *inputb);
  bool even_cg(void *resX, void *inputb);
  void generateOddB(void *new_b, void *tempVec1, void *tempVec2);
  void generateEvenB(void *new_b, void *b);

public:
  QcuCG(DSLASH_TYPE dslashType, CGParam *cgParam, double rsdTarget = 1e-10, int maxIterations = 1000,
        int blockSize = 256)
      : dslashType_(dslashType), cgParam_(cgParam), blockSize_(blockSize),
        innerProd_(QcuInnerProd(cgParam->msgHandler)), norm2_(QcuNorm2(cgParam->msgHandler)), dslashTempVec1_(nullptr),
        dslashTempVec2_(nullptr), numIterations_(0), maxIterations_(maxIterations), rsdTarget_(rsdTarget) {

    allocateTempVectors();

    if (dslashType_ == DSLASH_TYPE::DSLASH_WILSON) {

      dslashParam_ = new DslashParam(cgParam_->fermionInB, cgParam_->fermionOutX, cgParam_->gauge, cgParam_->Lx,
                                     cgParam_->Ly, cgParam_->Lz, cgParam_->Lt, EVEN_PARITY, cgParam_->Nx, cgParam_->Ny,
                                     cgParam_->Nz, cgParam_->Nt, cgParam_->kappa, QCU_DAGGER_NO, cgParam_->memPool,
                                     cgParam_->msgHandler, cgParam_->qcuComm, cgParam_->stream1, cgParam_->stream2);
      dslashParam_->tempFermionIn1 = dslashTempVec1_;
      dslashParam_->tempFermionIn2 = dslashTempVec2_;
#ifdef DEBUG
      printf("in function %s, line = %d, kappa = %lf, dslashParam_ = %p\n", __FUNCTION__, __LINE__, dslashParam_->kappa,
             dslashParam_);
#endif
      dslash_ = new WilsonDslash(dslashParam_, blockSize_);
      singleDslash_ = DslashMV(dslash_, blockSize_);
      cgIterMV_Odd_ = CGDslashMV_Odd(dslash_, blockSize_);

    } else {
      assert(0); // to be implemented
    }

    int Lx = cgParam_->Lx;
    int Ly = cgParam_->Ly;
    int Lz = cgParam_->Lz;
    int Lt = cgParam_->Lt;
    int subVol = Lx * Ly * Lz * Lt / 2;

    evenB_ = cgParam_->fermionInB;
    oddB_ = static_cast<void *>(static_cast<Complex *>(cgParam_->fermionInB) + subVol * Ns * Nc);
    evenX_ = cgParam_->fermionOutX;
    oddX_ = static_cast<void *>(static_cast<Complex *>(cgParam_->fermionOutX) + subVol * Ns * Nc);

#ifdef DEBUG
    double norm;
    norm2_(tmp1_, tmp2_, evenB_, cgParam_->Lx * cgParam_->Ly * cgParam_->Lz * cgParam_->Lt * Ns * Nc / 2,
           cgParam_->stream1);
    CHECK_CUDA(cudaMemcpyAsync(&norm, tmp1_, sizeof(double), cudaMemcpyDeviceToHost, cgParam_->stream1));
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
    printf("IN FILE qcu_cg.cu, FUNCTION qcuInvert, LINE %d, norm of input even b = %e. addr(evenB_) = %p\n", __LINE__,
           norm, evenB_);

    norm2_(tmp1_, tmp2_, oddB_, cgParam_->Lx * cgParam_->Ly * cgParam_->Lz * cgParam_->Lt * Ns * Nc / 2,
           cgParam_->stream1);
    CHECK_CUDA(cudaMemcpyAsync(&norm, tmp1_, sizeof(double), cudaMemcpyDeviceToHost, cgParam_->stream1));
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
    printf("IN FILE qcu_cg.cu, FUNCTION qcuInvert, LINE %d, norm of input odd b = %e. addr(oddB_) = %p\n", __LINE__,
           norm, oddB_);

    norm2_(tmp1_, tmp2_, evenB_, cgParam_->Lx * cgParam_->Ly * cgParam_->Lz * cgParam_->Lt * Ns * Nc,
           cgParam_->stream1);
    CHECK_CUDA(cudaMemcpyAsync(&norm, tmp1_, sizeof(double), cudaMemcpyDeviceToHost, cgParam_->stream1));
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
    printf("IN FILE qcu_cg.cu, FUNCTION qcuInvert, LINE %d, full norm of input b = %e. addr(evenB_) = %p\n", __LINE__,
           norm, evenB_);
#endif
  }

  // virtual void qcuInvert(void *resX, void *inputb);
  virtual void qcuInvert();

  virtual ~QcuCG() {
    freeTempVectors();
    if (dslash_ != nullptr) {
      delete dslash_;
    }
    if (dslashParam_ != nullptr) {
      delete dslashParam_;
    }
  }
};

END_NAMESPACE(qcu)