#pragma once

#include "qcd/qcu_dslash.cuh"

BEGIN_NAMESPACE(qcu)
class WilsonDslash : public Dslash {
  private:
  void preDslash(int dim, int dir, int daggerFlag = 0);
  void postDslash(int dim, int dir, int daggerFlag = 0);
  void dslashMpiIsendrecv(int dim);
  // void dslashMpiWait(int dim);
  void dslashMpiWait();

public:
  WilsonDslash(DslashParam *param, int blockSize = 256, cudaStream_t dslashStream1 = NULL,
               cudaStream_t dslashStream2 = NULL)
      : Dslash(param, blockSize, dslashStream1, dslashStream2) {}
  // use this function to call kernel function, this function donnot sync inside
  // virtual void apply(int daggerFlag = 0);                                            // to implement
  // virtual void preApply(int daggerFlag = 0); // to implement
  // virtual void postApply(int daggerFlag = 0);
  virtual void apply();                                            // to implement
  virtual void preApply(); // to implement
  virtual void postApply();
  // TODO: WILSON DSLASH MatMul
  void wilsonMatMul() {}
};

END_NAMESPACE(qcu)