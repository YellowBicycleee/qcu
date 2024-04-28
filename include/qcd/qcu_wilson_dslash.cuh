#pragma once

#include "qcd/qcu_dslash.cuh"

BEGIN_NAMESPACE(qcu)
class WilsonDslash : public Dslash {
 private:
  void preDslash(int dim, int dir, int daggerFlag = 0);
  void postDslash(int dim, int dir, int daggerFlag = 0);
  void preDslashMPI(int dim, int dir, int daggerFlag = 0);
  void postDslashMPI(int dim, int dir, int daggerFlag = 0);
  // void postDslashMPI(int dim, int dir, int daggerFlag = 0);
  void dslashNcclIsendrecv(int dim);
  void dslashMPIIsendrecv(int dim);
  void dslashNcclWait();
  void cudaStreamBarrier();
  void dslashMPIWait(int dim, int dir);

 public:
  WilsonDslash(DslashParam *param, int blockSize = 256) : Dslash(param, blockSize) {}
  virtual ~WilsonDslash() {}
  virtual void apply();     // to implement
  virtual void preApply();  // to implement
  virtual void postApply();

  virtual void preApply2();
  virtual void postApply2();
  // TODO: WILSON DSLASH MatMul
  void wilsonMatMul() {}
};

END_NAMESPACE(qcu)