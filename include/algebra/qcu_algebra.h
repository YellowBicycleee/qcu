#pragma once

#include "qcu_macro.cuh"

#define QCU_CUDA_ENABLED

#ifdef QCU_CUDA_ENABLED
#include <cuda.h>
#include <cuda_runtime.h>

BEGIN_NAMESPACE(qcu)

// template <int _dim>
// struct VectorDesc {
//   int dimLen[_dim];
//   /* data */
// };

struct QcuVectorAdd {
  typedef void *_genvector; // reserved for future use

  // template <typename _T>
  virtual _genvector operator()(_genvector result, _genvector operand1, _genvector operand2,
                                int vectorLength, cudaStream_t stream = NULL) = 0;
};

struct QcuGEMV {
  typedef void *_genvector;

  _genvector matrix;
  _genvector vector;

  // TODO: GEMV
  virtual void operator()(_genvector result, _genvector operand1, _genvector operand2,
                                int parity = 0, cudaStream_t stream = NULL) {
    printf("GEMV not implemented\n");
  };
};

#endif

END_NAMESPACE(qcu)