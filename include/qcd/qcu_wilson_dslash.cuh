#pragma once

#include "qcd/qcu_dslash.cuh"


static void syncEvent(cudaEvent_t event) {
  CHECK_CUDA(cudaEventRecord(event, 0));
  CHECK_CUDA(cudaEventSynchronize(event));
}

BEGIN_NAMESPACE(qcu)
class WilsonDslash : public Dslash {
 private:
  void preDslashMPI(int dim, int dir, int daggerFlag = 0);
  void postDslashMPI(int dim, int dir, int daggerFlag = 0);
  void dslashMPIIsendrecv(int dim);
  void cudaStreamBarrier();
  void dslashMPIWait(int dim, int dir);

 public:
  WilsonDslash(DslashParam *param, int blockSize = 256) : Dslash(param, blockSize) {}
  virtual ~WilsonDslash() {}
  virtual void apply();     // to implement

  virtual void preApply2();
  virtual void postApply2();
  // TODO: WILSON DSLASH MatMul
  void wilsonMatMul() {}
};

END_NAMESPACE(qcu)