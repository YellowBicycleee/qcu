#pragma once
#include "qcd_macro.cuh"
#include "qcd/qcu_dslash.cuh"


BEGIN_NAMESPACE(qcu)

class QcuCG {
  Dslash *dslash;
public:
  virtual void operator()() = 0;
};

END_NAMESPACE(qcu)