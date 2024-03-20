#include "qcd/qcu_wilson_dslash.cuh"
#include "qcu.h"
#include "qcu_macro.cuh"
#include "qcu_storage/qcu_storage.cuh"
#include <cuda.h>

#define PRINT_EXEC_TIME

BEGIN_NAMESPACE(qcu)
enum DslashType { QCU_DSLASH_WILSON = 0, QCU_DSLASH_CLOVER = 1 };

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

  cudaEvent_t startEvent_;
  cudaEvent_t stopEvent_;

public:
  Qcu(int Lx, int Ly, int Lz, int Lt, int Nx, int Ny, int Nz, int Nt, double mass = 0.0)
      : Lx_(Lx), Ly_(Ly), Lz_(Lz), Lt_(Lt), procNx_(Nx), procNy_(Ny), procNz_(Nz), procNt_(Nt),
        mass_(mass), kappa_(1.0 / (2.0 * (4.0 + mass))), gaugeLoaded_(false), inputGauge_(nullptr),
        coalescedGauge_(nullptr), coalescedFermionIn_(nullptr), coalescedFermionOut_(nullptr),
        fermionIn_(nullptr), fermionOut_(nullptr), cloverMatrix_(nullptr),
        cloverInvMatrix_(nullptr) {
    CHECK_CUDA(cudaStreamCreate(&stream1_));
    CHECK_CUDA(cudaStreamCreate(&stream2_));
    CHECK_CUDA(cudaEventCreate(&startEvent_));
    CHECK_CUDA(cudaEventCreate(&stopEvent_));
    int vol = Lx_ * Ly_ * Lz_ * Lt_ / 2;
    CHECK_CUDA(cudaMalloc(&coalescedFermionIn_, sizeof(double) * vol * 2 * Ns * Nc));
    CHECK_CUDA(cudaMalloc(&coalescedFermionOut_, sizeof(double) * vol * 2 * Ns * Nc));
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
  }

  // TODO : load gauge
  void loadGauge(void *gauge);
  void shiftFermionStorage(void *dst, void *src, int shiftDir);
  // TODO : modify lattice size
  void modifyLattice(int Lx, int Ly, int Lz, int Lt) {}

  // TODO : dslash wilson
  virtual void wilsonDslash(void *fermionOut, void *fermionIn, int parity);
  virtual void wilsonMatMul() {}
  // TODO : dslash clover
  virtual void cloverDslash() {}
  virtual void cloverMatMul() {}
  // TODO : invert (cg inverter)
  virtual void qcuInvert() {}
};

void Qcu::wilsonDslash(void *fermionOut, void *fermionIn, int parity) {
  fermionIn_ = fermionIn;
  fermionOut_ = fermionOut;
  shiftFermionStorage(coalescedFermionIn_, fermionIn_, TO_COALESCE);

  DslashParam dslashParam(coalescedFermionIn_, coalescedFermionOut_, coalescedGauge_, Lx_, Ly_, Lz_,
                          Lt_, parity, procNx_, procNy_, procNz_, procNt_);

  CHECK_CUDA(cudaEventRecord(startEvent_, stream1_));
  WilsonDslash dslash(&dslashParam, 256, stream1_);
  dslash.apply(0);
  CHECK_CUDA(cudaEventRecord(stopEvent_, stream1_));
  CHECK_CUDA(cudaEventSynchronize(stopEvent_));

#ifdef PRINT_EXEC_TIME
  float elapsedTime;
  CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, startEvent_, stopEvent_));
  printf("Recorded time : %f s\n", elapsedTime / 1000);
#endif

  shiftFermionStorage(fermionOut_, coalescedFermionOut_, TO_NON_COALESCE);
}

void Qcu::loadGauge(void *gauge) {
  if (!gaugeLoaded_ && coalescedGauge_ == nullptr) {
    CHECK_CUDA(cudaMalloc(&coalescedGauge_,
                          sizeof(double) * Nd * Lx_ * Ly_ * Lz_ * Lt_ * (Nc - 1) * Nc * 2));
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
END_NAMESPACE(qcu)

static qcu::Qcu *qcu_ptr = nullptr;

void initGridSize(QcuGrid_t *grid, QcuParam *p_param, void *gauge, void *fermion_in,
                  void *fermion_out) {
  if (qcu_ptr == nullptr) {
    qcu_ptr = new qcu::Qcu(p_param->lattice_size[0], p_param->lattice_size[1],
                           p_param->lattice_size[2], p_param->lattice_size[3], grid->grid_size[0],
                           grid->grid_size[1], grid->grid_size[2], grid->grid_size[3]);
  }
}

void destroyQcu() {
  if (qcu_ptr != nullptr) {
    delete qcu_ptr;
    qcu_ptr = nullptr;
  }
}

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity) {
  qcu_ptr->loadGauge(gauge);
  qcu_ptr->wilsonDslash(fermion_out, fermion_in, parity);
}

void fullDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param,
                   int dagger_flag) {}
void cg_inverter(void *x_vector, void *b_vector, void *gauge, QcuParam *param, double p_max_prec,
                 double p_kappa) {}

// TODO : delete parameter param.
void loadQcuGauge(void *gauge, QcuParam *param) {}