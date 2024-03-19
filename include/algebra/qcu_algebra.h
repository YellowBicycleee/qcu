#pragma once

#include "qcu_macro.cuh"

#define QCU_CUDA_ENABLED

#ifdef QCU_CUDA_ENABLED
#include <cuda.h>
#include <cuda_runtime.h>


BEDIN_NAMESPACE(qcu)

struct QcuMatAdd {
  typedef void *_genvector;

public:
  virtual _genvector operator()(_genvector operand1, _genvector operand2,
                                _genvector result, cudaStream_t& stream = NULL) = 0;
};

struct QcuMatMul {
  typedef void *_genvector;

public:
  virtual _genvector operator()(_genvector operand1, _genvector operand2,
                                _genvector result, cudaStream_t& stream = NULL) = 0;
};

struct QcuVectorAdd : public QcuMatrixAdd {
private:
  const int vectorLength;
  void addKernel() { // TODO: add kernel
    // call gpu kernel functions, and MPI comms
  }

public:
  QcuVectorAdd(int vectorLength) : vectorLength(vectorLength) {}

  virtual _genvector operator()(_genvector operand1, _genvector operand2,
                                _genvector result, cudaStream_t& stream = NULL) // override
  {
    addKernel();
  }
};
#endif
END_NAMESPACE(qcu)