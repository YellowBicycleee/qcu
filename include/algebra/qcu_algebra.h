#pragma once

#include "basic_data/qcu_complex.cuh"
#include "comm/qcu_communicator.h"
#include "qcu_macro.cuh"
#define QCU_CUDA_ENABLED

#ifdef QCU_CUDA_ENABLED
#include <cuda.h>
#include <cuda_runtime.h>

BEGIN_NAMESPACE(qcu)

struct QcuVectorAdd {
  typedef void *_genvector; // reserved for future use, more convinient for template
  int blockSize;

  QcuVectorAdd(int pBlockSize = 256) : blockSize(pBlockSize) {}
  // template <typename _T>
  virtual void operator()(_genvector result, _genvector operand1, _genvector operand2,
                          int vectorLength, cudaStream_t stream = NULL);
};

// FUNCTOR: QcuSPMV
// 用于计算矩阵operand1和向量operand2的乘积，最终结果放在result
// GESPMV暂时不实现，在dslash头文件中继承functor实现特殊的SPMV
struct QcuSPMV {
  typedef void *_genvector;
  int blockSize;
  MsgHandler *msgHandler;
  _genvector matrix;
  _genvector vector;

  QcuSPMV(MsgHandler *pMsgHandler, int pBlockSize = 256)
      : msgHandler(pMsgHandler), blockSize(pBlockSize) {}

  // general SPMV（稀疏矩阵向量乘）暂时不实现
  virtual void operator()(_genvector result, _genvector operand1, _genvector operand2,
                          int parity = 0, cudaStream_t stream = NULL) {
    printf("SPMV not implemented\n");
  };
};

/// FUNCTOR: QcuInnerProd
/// 用于计算向量operand1和operand2的内积，最终结果放在result，中间结果放在temp_result
/// 待优化：单进程情况下，理论上不需要reduce，（但是会变得ugly）
struct QcuInnerProd {
  typedef void *_genvector;
  int blockSize;
  MsgHandler *msgHandler;
  // template <typename _T>
  QcuInnerProd(MsgHandler *pMsgHandler = nullptr, int pBlockSize = 256)
      : msgHandler(pMsgHandler), blockSize(pBlockSize) {}
  virtual void operator()(_genvector result, _genvector temp_result, _genvector operand1,
                          _genvector operand2, int vectorLength, cudaStream_t stream = NULL);
};

/// FUNCTOR: QcuNorm2
/// 用于计算向量operand的norm2，最终结果放在result，中间结果放在temp_result
/// 待优化：单进程情况下，理论上不需要reduce，（但是会变得ugly）
struct QcuNorm2 {
  typedef void *_genvector;
  int blockSize;
  MsgHandler *msgHandler;
  // template <typename _T>
  QcuNorm2(MsgHandler *pMsgHandler, int pBlockSize = 256)
      : msgHandler(pMsgHandler), blockSize(pBlockSize) {}
  virtual void operator()(_genvector result, _genvector temp_result, _genvector operand,
                          int vectorLength, cudaStream_t stream = NULL);
};

// result = alpha * operand1 + operand2
struct QcuSaxpy {
  typedef void *_genvector;
  int blockSize;

  QcuSaxpy(int pBlockSize = 256) : blockSize(pBlockSize) {}
  virtual void operator()(_genvector result, Complex alpha, _genvector operandX,
                          _genvector operandY, int vectorLength, cudaStream_t stream = NULL);
};

// result = alpha * operand
struct QcuSax {
  typedef void *_genvector;
  int blockSize;

  QcuSax(int pBlockSize = 256) : blockSize(pBlockSize) {}
  virtual void operator()(_genvector result, Complex alpha, _genvector operandX, int vectorLength,
                          cudaStream_t stream = NULL);
};

// result = operand
struct QcuComplexCopy {
  typedef void *_genvector;
  int blockSize;

  QcuComplexCopy(int pBlockSize = 256) : blockSize(pBlockSize) {}
  virtual void operator()(_genvector result, _genvector operand, int vectorLength,
                          cudaStream_t stream = NULL);
};
void complexDivideGPU(void *res, void *a, void *b, cudaStream_t stream = NULL);
#endif

END_NAMESPACE(qcu)