#pragma once

#include "qcd/qcu_dslash.cuh"

BEGIN_NAMESPACE(qcu)
class WilsonDslash : public Dslash {
private:
  void preDslash(int dim, int dir, int daggerFlag = 0);
  void postDslash(int dim, int dir, int daggerFlag = 0);
  void dslashMpiIsendrecv(int dim);
  void dslashMpiWait();

public:
  WilsonDslash(DslashParam *param, int blockSize = 256) : Dslash(param, blockSize) {}
  virtual ~WilsonDslash() {}
  virtual void apply();    // to implement
  virtual void preApply(); // to implement
  virtual void postApply();
  // TODO: WILSON DSLASH MatMul
  void wilsonMatMul() {}
};

END_NAMESPACE(qcu)