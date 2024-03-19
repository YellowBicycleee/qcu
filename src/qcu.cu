#include "interface/qcu.h"
#include "qcd/qcu_wilson_dslash.cuh"
#include "qcu_macro.h"
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
  void *coalescedGauge_;
  void *fermionIn_;
  void *fermionOut_;

  void *cloverMatrix_;
  void *cloverInvMatrix_;

  cudaStream_t stream1_;
  cudaStream_t stream2_;

  cudaEvent_t startEvent_;
  cudaEvent_t stopEvent_;

public:
  Qcu(int Lx, int Ly, int Lz, int Lt, int Nx, int Ny, int Nz, int Nt, double mass = 0.0)
      : Lx_(Lx), Ly_(Ly), Lz_(Lz), Lt_(Lt), procNx_(Nx), procNy_(Ny), procNz_(Nz), procNt_(Nt),
        mass_(mass), kappa_(1.0 / (2.0 * (4.0 + mass_))), gaugeLoaded_(false), inputGauge_(nullptr),
        coalescedGauge_(nullptr), fermionIn_(nullptr), fermionOut_(nullptr), cloverMatrix_(nullptr),
        cloverInvMatrix_(nullptr) {
    CHECK_CUDA(cudaStreamCreate(&stream1_));
    CHECK_CUDA(cudaStreamCreate(&stream2_));
    CHECK_CUDA(cudaEventCreate(&startEvent_));
    CHECK_CUDA(cudaEventCreate(&stopEvent_));
  }
  void virtual ~Qcu() {
    CHECK_CUDA(cudaStreamDestroy(stream1_));
    CHECK_CUDA(cudaStreamDestroy(stream2_));
    CHECK_CUDA(cudaEventDestroy(startEvent_));
    CHECK_CUDA(cudaEventDestroy(stopEvent_));
  }
  // TODO : load gauge
  void loadGauge(void *gauge) {}
  // TODO : modify lattice size
  void modifyLattice(int Lx, int Ly, int Lz, int Lt) {}

  // TODO : dslash wilson
  virtual void wilsonDslash(void *fermionOut, void *fermionIn);
  virtual void wilsonMatMul() {}
  // TODO : dslash clover
  virtual void cloverDslash() {}
  virtual void cloverMatMul() {}
  // TODO : invert (cg inverter)
  virtual void qcuInvert() {}
};

void Qcu::wilsonDslash(void *fermionOut, void *fermionIn) {

  DslashParam dslashParam(fermionIn, fermionOut, coalescedGauge_, Lx_, Ly_, Lz_, Lt_, int p_parity);

  CHECK_CUDA(cudaEventRecord(startEvent_, stream1_));
  WilsonDslash wilsonDslash(&dslashParam, 256, stream1_);
  wilsonDslash.apply(0);
  CHECK_CUDA(cudaEventRecord(stopEvent_, stream1_));
  CHECK_CUDA(cudaEventSynchronize(stopEvent_));

#ifdef PRINT_EXEC_TIME
  float elapsedTime;
  CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, startEvent_, stopEvent_));
  printf("Recorded time : %f ms\n", elapsedTime);
#endif
}

void Qcu::loadGauge(void *gauge) {
  if (!gaugeLoaded_ && coalescedGauge_ == nullptr) {
    CHECK_CUDA(cudaMalloc(&coalescedGauge_,
                          sizeof(double) * Nd * Lx_ * Ly_ * Lz_ * Lt_ * (Nc - 1) * Nc * 2));
  }

//   shiftGaugeStorageTwoDouble(qcu_gauge, gauge, TO_COALESCE, Lx, Ly, Lz, Lt);
  gaugeLoaded_ = true;
}
END_NAMESPACE(qcu)

static qcu::Qcu *qcu_ptr = nullptr;

void initGridSize(QcuGrid_t *grid, QcuParam *p_param, void *gauge, void *fermion_in,
                  void *fermion_out) {
  if (qcu_ptr == nullptr) {
    qcu_ptr = new Qcu(p_param->lattice_size[0], p_param->lattice_size[1], p_param->latice_size[2],
                      p_param->lattice_size[3], grid->grid_size[0], grid->grid_size[1],
                      grid->grid_size[2], grid->grid_size[3]);
  }
}

void destroyQcu() {
  if (qcu_ptr != nullptr) {
    delete qcu_ptr;
    qcu_ptr = nullptr;
  }
}

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity) {}

void fullDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param,
                   int dagger_flag) {}
void cg_inverter(void *x_vector, void *b_vector, void *gauge, QcuParam *param, double p_max_prec,
                 double p_kappa) {}

// TODO : delete parameter param.
void loadQcuGauge(void *gauge, QcuParam *param) {

}

void loadQcuGauge(void *gauge, QcuParam *param) {
  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];

  checkCudaErrors(
      cudaMalloc(&qcu_gauge, sizeof(double) * Nd * Lx * Ly * Lz * Lt * (Nc - 1) * Nc * 2));
  shiftGaugeStorageTwoDouble(qcu_gauge, gauge, TO_COALESCE, Lx, Ly, Lz, Lt);
  mpi_comm->setCoalescedGauge(qcu_gauge);
}