#include "algebra/qcu_algebra.h"
#include "qcu_macro.cuh"
#include "targets/qcu_linear_algebra.cuh"

BEGIN_NAMESPACE(qcu)
typedef void *_genvector;

void QcuInnerProd::operator()(_genvector result, _genvector temp_result, _genvector operand1,
                              _genvector operand2, int vectorLength, cudaStream_t stream) {
  int gridSize = (vectorLength + blockSize - 1) / blockSize;
  innerProduct<<<gridSize, blockSize, blockSize, stream>>>(result, operand1, operand2,
                                                           vectorLength);
  // reduce result
  complexReduceSum<<<1, blockSize, blockSize, stream>>>(temp_result, result, gridSize);
  ncclAllReduce(temp_result, result, 2, ncclDouble, ncclSum, msgHandler->ncclComm, stream);
}

void QcuVectorAdd::operator()(_genvector result, _genvector operand1, _genvector operand2,
                              int vectorLength, cudaStream_t stream) {
  int gridSize = (vectorLength + blockSize - 1) / blockSize;
  double2VectorAdd<<<gridSize, blockSize, blockSize, stream>>>(result, operand1, operand2,
                                                               vectorLength);
}

// norm2
void QcuNorm2::operator()(_genvector result, _genvector temp_result, _genvector operand,
                          int vectorLength, cudaStream_t stream) {
  int gridSize = (vectorLength + blockSize - 1) / blockSize;
  // 第三个参数是字节大小
  norm2Square<<<gridSize, blockSize, blockSize * sizeof(double), stream>>>(temp_result, operand, vectorLength);
  // printf("QcuNorm2:: %d %d\n", gridSize, blockSize);
  doubleReduceSum<<<1, blockSize, blockSize * sizeof(double), stream>>>(temp_result, temp_result, gridSize);

  ncclAllReduce(temp_result, result, 1, ncclDouble, ncclSum, msgHandler->ncclComm, stream);
  doubleSqrt<<<1, 1, 0, stream>>>(result, result); // sqrt
}

void complexDivideGPU(void *res, void *a, void *b, cudaStream_t stream) {
  complexDivideKernel<<<1, 1, 0, stream>>>(res, a, b);
}

END_NAMESPACE(qcu)