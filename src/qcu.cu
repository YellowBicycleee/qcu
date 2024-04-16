#include "comm/qcu_communicator.h"
#include "mempool/qcu_mempool.h"
#include "qcd/qcu_wilson_dslash.cuh"
#include "qcu.h"
#include "qcu_macro.cuh"
#include "qcu_storage/qcu_storage.cuh"
#include "solver/qcu_cg.h"
#include <cuda.h>
// #define PRINT_EXEC_TIME
// #define PRINT_ALLOCATED_MEM_SIZE

BEGIN_NAMESPACE(qcu)

class Qcu {
protected:
  bool gaugeLoaded_;

  double kappa_;
  double mass_;

  int Lx_;
  int Ly_;
  int Lz_;
  int Lt_;

  int procNx_;
  int procNy_;
  int procNz_;
  int procNt_;
  int boundaryLength_[Nd];

  void *inputGauge_;

  void *coalescedGauge_;      // coalesced gauge
  void *coalescedFermionIn_;  // coalesced fermion
  void *coalescedFermionOut_; // coalesced fermion

  void *fermionIn_;  // input fermion
  void *fermionOut_; // output fermion

  void *cloverMatrix_;
  void *cloverInvMatrix_;

  cudaStream_t stream1_;
  cudaStream_t stream2_;
  cudaStream_t commStreams_[Nd * DIRECTIONS];
  cudaEvent_t startEvent_;
  cudaEvent_t stopEvent_;

  QcuMemPool *memPool_;
  MsgHandler *msgHandler_;
  QcuComm *qcuComm_;

public:
  Qcu(int Lx, int Ly, int Lz, int Lt, int Nx, int Ny, int Nz, int Nt, double mass = 0.0)
      : Lx_(Lx), Ly_(Ly), Lz_(Lz), Lt_(Lt), procNx_(Nx), procNy_(Ny), procNz_(Nz), procNt_(Nt), mass_(mass),
        kappa_(1.0 / (2.0 * (4.0 + mass))), gaugeLoaded_(false), inputGauge_(nullptr), coalescedGauge_(nullptr),
        coalescedFermionIn_(nullptr), coalescedFermionOut_(nullptr), fermionIn_(nullptr), fermionOut_(nullptr),
        cloverMatrix_(nullptr), cloverInvMatrix_(nullptr), memPool_(nullptr), msgHandler_(nullptr), qcuComm_(nullptr) {
    CHECK_CUDA(cudaStreamCreate(&stream1_));
    CHECK_CUDA(cudaStreamCreate(&stream2_));
    for(int i = 0; i < Nd * DIRECTIONS; i++) {
      CHECK_CUDA(cudaStreamCreate(&commStreams_[i]));
    }

    CHECK_CUDA(cudaEventCreate(&startEvent_));
    CHECK_CUDA(cudaEventCreate(&stopEvent_));
    int vol = Lx_ * Ly_ * Lz_ * Lt_;

    // TODO：此部分分配内存过剩，看一下怎么减少
    CHECK_CUDA(cudaMalloc(&coalescedFermionIn_, sizeof(double) * 2 * vol * Ns * Nc));
    CHECK_CUDA(cudaMalloc(&coalescedFermionOut_, sizeof(double) * 2 * vol * Ns * Nc));
    msgHandler_ = new MsgHandler();
    qcuComm_ = new QcuComm(procNx_, procNy_, procNz_, procNt_);
    memPoolInit();
#ifdef DEBUG
    printf("Qcu mass = %lf, kappa_ = %lf\n", mass_, kappa_);
#endif
  }
  virtual ~Qcu() {
    CHECK_CUDA(cudaStreamDestroy(stream1_));
    CHECK_CUDA(cudaStreamDestroy(stream2_));
    CHECK_CUDA(cudaEventDestroy(startEvent_));
    CHECK_CUDA(cudaEventDestroy(stopEvent_));

    if (coalescedGauge_ != nullptr) {
      CHECK_CUDA(cudaFree(coalescedGauge_));
      coalescedGauge_ = nullptr;
    }
    if (coalescedFermionIn_ != nullptr) {
      CHECK_CUDA(cudaFree(coalescedFermionIn_));
      coalescedFermionIn_ = nullptr;
    }
    if (coalescedFermionOut_ != nullptr) {
      CHECK_CUDA(cudaFree(coalescedFermionOut_));
      coalescedFermionOut_ = nullptr;
    }
    if (memPool_ != nullptr) {
      delete memPool_;
      memPool_ = nullptr;
    }
    if (msgHandler_ != nullptr) {
      delete msgHandler_;
      msgHandler_ = nullptr;
    }
    if (qcuComm_ != nullptr) {
      delete qcuComm_;
      qcuComm_ = nullptr;
    }
  }
  void memPoolInit() {
    memPool_ = new QcuMemPool();
    int singleVecLength = Ns * Nc;

    boundaryLength_[0] = procNx_ == 1 ? 0 : Ly_ * Lz_ * Lt_ / 2 * singleVecLength;
    boundaryLength_[1] = procNy_ == 1 ? 0 : Lx_ * Lz_ * Lt_ / 2 * singleVecLength;
    boundaryLength_[2] = procNz_ == 1 ? 0 : Lx_ * Ly_ * Lt_ / 2 * singleVecLength;
    boundaryLength_[3] = procNt_ == 1 ? 0 : Lx_ * Ly_ * Lz_ / 2 * singleVecLength;

    memPool_->allocateAllVector(boundaryLength_[0], boundaryLength_[1], boundaryLength_[2], boundaryLength_[3],
                                sizeof(double) * 2);
#ifdef PRINT_ALLOCATED_MEM_SIZE
    printf("========================\n");
    printf("Allocated memory size : \n x dim = %d\n y dim = %d\n z dim = %d\n t dim = %d\n ", boundaryLength_[0],
           boundaryLength_[1], boundaryLength_[2], boundaryLength_[3]);
    printf("========================\n");
#endif
  }

  void loadGauge(void *gauge);
  void shiftFermionStorage(void *dst, void *src, int shiftDir);

  // virtual void wilsonDslash(void *fermionOut, void *fermionIn, int parity);
  virtual void wilsonDslashMultiProc(void *fermionOut, void *fermionIn, int parity);
  // virtual void wilsonMatMul() {}
  // TODO : dslash clover
  // virtual void cloverDslash() {}
  // virtual void cloverMatMul() {}
  // TODO : invert (cg inverter)
  virtual void qcuInvert(void *fermionOutX, void *fermionInB, double diffTarget, int maxIterations);
};

void Qcu::wilsonDslashMultiProc(void *fermionOut, void *fermionIn, int parity) {

  // shiftStorage
  int daggerFlag = 0;
  fermionIn_ = fermionIn;
  fermionOut_ = fermionOut;
  shiftFermionStorage(coalescedFermionIn_, fermionIn_, TO_COALESCE);

  DslashParam dslashParam(coalescedFermionIn_, coalescedFermionOut_, coalescedGauge_, Lx_, Ly_, Lz_, Lt_, parity,
                          procNx_, procNy_, procNz_, procNt_, kappa_, daggerFlag, memPool_, msgHandler_, qcuComm_,
                          stream1_, stream2_, commStreams_);
  CHECK_CUDA(cudaEventRecord(startEvent_, stream1_));
  WilsonDslash dslash(&dslashParam, 256);

  // dslash.preApply();
  // dslash.apply();
  // dslash.postApply();
  dslash.preApply2();
  dslash.apply();
  dslash.postApply2();

  // DslashMV dslashMv(&dslash);
  // dslashMv(coalescedFermionOut_, coalescedFermionIn_);
  // CHECK_CUDA(cudaEventRecord(stopEvent_, stream1_));
  // CHECK_CUDA(cudaEventSynchronize(stopEvent_));
#ifdef PRINT_EXEC_TIME
  float elapsedTime;
  CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, startEvent_, stopEvent_));
  printf("Recorded time : %f s\n", elapsedTime / 1000);
#endif
  // shiftStorage
  shiftFermionStorage(fermionOut_, coalescedFermionOut_, TO_NON_COALESCE);
}

void Qcu::loadGauge(void *gauge) {
  if (!gaugeLoaded_ && coalescedGauge_ == nullptr) {
    CHECK_CUDA(cudaMalloc(&coalescedGauge_, sizeof(double) * Nd * Lx_ * Ly_ * Lz_ * Lt_ * (Nc - 1) * Nc * 2));
  }

  shiftGaugeStorageTwoDouble(coalescedGauge_, gauge, TO_COALESCE, Lx_, Ly_, Lz_, Lt_);
  gaugeLoaded_ = true;
}

// TODO : 消除不必要代码
void Qcu::shiftFermionStorage(void *dst, void *src, int shiftDir) {
  if (shiftDir == TO_COALESCE) {
    shiftVectorStorageTwoDouble(dst, src, TO_COALESCE, Lx_, Ly_, Lz_, Lt_);
  } else if (shiftDir == TO_NON_COALESCE) {
    shiftVectorStorageTwoDouble(dst, src, TO_NON_COALESCE, Lx_, Ly_, Lz_, Lt_);
  }
}

void Qcu::qcuInvert(void *fermionOutX, void *fermionInB, double diffTarget, int maxIterations) {
  int vol = Lx_ * Ly_ * Lz_ * Lt_;
  int halfVol = vol / 2;
  void *originFermionInEven = fermionInB;
  void *originFermionInOdd = static_cast<void *>(static_cast<Complex *>(fermionInB) + halfVol * Ns * Nc);
  void *originFermionOutEven = fermionOutX;
  void *originFermionOutOdd = static_cast<void *>(static_cast<Complex *>(fermionOutX) + halfVol * Ns * Nc);

  void *newFermionInEven = coalescedFermionIn_;
  void *newFermionInOdd = static_cast<void *>(static_cast<Complex *>(coalescedFermionIn_) + halfVol * Ns * Nc);
  void *newFermionOutEven = coalescedFermionOut_;
  void *newFermionOutOdd = static_cast<void *>(static_cast<Complex *>(coalescedFermionOut_) + halfVol * Ns * Nc);

  // shift storage to coalesce
  shiftFermionStorage(newFermionInEven, originFermionInEven, TO_COALESCE);
  shiftFermionStorage(newFermionInOdd, originFermionInOdd, TO_COALESCE);
#ifdef DEBUG
  QcuNorm2 norm2(msgHandler_, 256);
  void *temp1;
  void *temp2;
  double norm;
  CHECK_CUDA(cudaMalloc(&temp1, sizeof(Complex) * vol * Ns * Nc));
  CHECK_CUDA(cudaMalloc(&temp2, sizeof(Complex) * vol * Ns * Nc));
  norm2(temp1, temp2, fermionInB, vol * Ns * Nc);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(&norm, temp1, sizeof(double), cudaMemcpyDeviceToHost));
  printf("__FILE__ = %s, __LINE__ = %d, norm = %lf, addr(b) = %p\n", __FILE__, __LINE__, norm, fermionInB);
  norm2(temp1, temp2, newFermionInEven, vol * Ns * Nc);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(&norm, temp1, sizeof(double), cudaMemcpyDeviceToHost));
  printf("__FILE__ = %s, __LINE__ = %d, coalesced norm = %lf, addr(newB) = %p\n", __FILE__, __LINE__, norm,
         newFermionInEven);
#endif

  // CG Inverter
  // generate CGParam
  assert(coalescedGauge_ != nullptr);
  CGParam cgParam(coalescedFermionIn_, coalescedFermionOut_, coalescedGauge_, nullptr, nullptr, kappa_, Lx_, Ly_, Lz_,
                  Lt_, procNx_, procNy_, procNz_, procNt_, memPool_, msgHandler_, qcuComm_, stream1_, stream2_);

  QcuCG qcuWilsonSolver_CG(DSLASH_WILSON, &cgParam, diffTarget, maxIterations, 256);
  qcuWilsonSolver_CG.qcuInvert();
  // shift back
  shiftFermionStorage(originFermionOutEven, newFermionOutEven, TO_NON_COALESCE);
  shiftFermionStorage(originFermionOutOdd, newFermionOutOdd, TO_NON_COALESCE);
}

END_NAMESPACE(qcu)

static qcu::Qcu *qcu_ptr = nullptr;

void initGridSize(QcuGrid_t *grid, QcuParam *p_param, void *gauge, void *fermion_in, void *fermion_out) {
  if (qcu_ptr == nullptr) {
    qcu_ptr = new qcu::Qcu(p_param->lattice_size[0], p_param->lattice_size[1], p_param->lattice_size[2],
                           p_param->lattice_size[3], grid->grid_size[0], grid->grid_size[1], grid->grid_size[2],
                           grid->grid_size[3]);
  }
}

void destroyQcu() {
  if (qcu_ptr != nullptr) {
    delete qcu_ptr;
    qcu_ptr = nullptr;
  }
}

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity) {
  // qcu_ptr->loadGauge(gauge);
  qcu_ptr->wilsonDslashMultiProc(fermion_out, fermion_in, parity);
}

void fullDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int dagger_flag) {}
void cg_inverter(void *x_vector, void *b_vector, void *gauge, QcuParam *param, double p_max_prec, double p_kappa) {
  // p_kappa is useless
  // qcu_ptr->qcuInvert(x_vector, b_vector, p_max_prec);
  qcu_ptr->qcuInvert(x_vector, b_vector, p_max_prec, 1000);
}

void loadQcuGauge(void *gauge, QcuParam *param) { qcu_ptr->loadGauge(gauge); }