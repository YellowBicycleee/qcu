#pragma once

#include "qcd/qcu_dslash.cuh"

BEGIN_NAMESPACE(qcu)
class WilsonDslash : public Dslash {

public:
  WilsonDslash(DslashParam* param, int blockSize = 256, cudaStream_t dslashStream = NULL)
      : Dslash(param, blockSize, dslashStream) {}
  // use this function to call kernel function, this function donnot sync inside
  virtual void apply(int daggerFlag = 0); // to implement

  // TODO: WILSON DSLASH MatMul
  void wilsonMatMul() {}
};

END_NAMESPACE(qcu)