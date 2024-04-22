#include "algebra/qcu_algebra.h"
#include "qcu_macro.cuh"
#include "targets/qcu_linear_algebra.cuh"

BEGIN_NAMESPACE(qcu)
typedef void *_genvector;
// #define DISABLE_CUDA

#ifdef QCU_CUDA_ENABLED

enum COMM_TYPE { NCCL_COMM, MPI_COMM, MPI_CONTINUOUS_COMM };

// template <>

static COMM_TYPE reduce_type = MPI_COMM;

void QcuInnerProd::operator()(_genvector result, _genvector temp_result, _genvector operand1, _genvector operand2,
                              int vectorLength, cudaStream_t stream) {
  int gridSize = (vectorLength + blockSize - 1) / blockSize / 2;
  innerProduct<<<gridSize, blockSize, blockSize * sizeof(double) * 2, stream>>>(temp_result, operand1, operand2,
                                                                                vectorLength);
  // reduce result
  complexReduceSum<<<1, blockSize, blockSize * sizeof(double) * 2, stream>>>(temp_result, temp_result, gridSize);
  // interprocess reduce
  if (reduce_type == NCCL_COMM) {
    ncclAllReduce(temp_result, result, 2, ncclDouble, ncclSum, msgHandler->ncclComm, stream);
  } else if (reduce_type == MPI_COMM) {
    Complex temp_res;
    Complex reduce_res;
    CHECK_CUDA(cudaMemcpyAsync(&temp_res, temp_result, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MPI_Allreduce(&temp_res, &reduce_res, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    CHECK_CUDA(cudaMemcpyAsync(result, &reduce_res, 2 * sizeof(double), cudaMemcpyHostToDevice, stream));
  } else if (reduce_type == MPI_CONTINUOUS_COMM) {

  } else {
    assert(0);
  }
}

void QcuVectorAdd::operator()(_genvector result, _genvector operand1, _genvector operand2, int vectorLength,
                              cudaStream_t stream) {
#ifndef DISABLE_CUDA
  int gridSize = (vectorLength + blockSize - 1) / blockSize;
  double2VectorAdd<<<gridSize, blockSize, blockSize, stream>>>(result, operand1, operand2, vectorLength);
#else
  Complex *result_ = (Complex *)result;
  Complex *operand1_ = (Complex *)operand1;
  Complex *operand2_ = (Complex *)operand2;
  for (int i = 0; i < vectorLength; i++) {
    result_[i] = operand1_[i] + operand2_[i];
  }
#endif
}

// norm2
void QcuNorm2::operator()(_genvector result, _genvector temp_result, _genvector operand, int vectorLength,
                          cudaStream_t stream) {
  int gridSize = (vectorLength + blockSize - 1) / blockSize / 2;
  // 第三个参数是字节大小
  norm2Square<<<gridSize, blockSize, blockSize * sizeof(double) * 2, stream>>>(temp_result, operand, vectorLength);
  // printf("QcuNorm2:: %d %d\n", gridSize, blockSize);
  doubleReduceSum<<<1, blockSize, blockSize * sizeof(double), stream>>>(temp_result, temp_result, gridSize);

  // ncclAllReduce(temp_result, result, 1, ncclDouble, ncclSum, msgHandler->ncclComm, stream);
    // interprocess reduce
  if (reduce_type == NCCL_COMM) {
    ncclAllReduce(temp_result, result, 1, ncclDouble, ncclSum, msgHandler->ncclComm, stream);
  } else if (reduce_type == MPI_COMM) {
    double temp_res;
    double reduce_res;
    CHECK_CUDA(cudaMemcpyAsync(&temp_res, temp_result, sizeof(double), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MPI_Allreduce(&temp_res, &reduce_res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    CHECK_CUDA(cudaMemcpyAsync(result, &reduce_res, sizeof(double), cudaMemcpyHostToDevice, stream));
  } else if (reduce_type == MPI_CONTINUOUS_COMM) {

  } else {
    assert(0);
  }
  doubleSqrt<<<1, 1, 0, stream>>>(result, result);  // sqrt
}

// result = alpha * operand1 + operand2
void QcuSaxpy::operator()(_genvector result, Complex alpha, _genvector operandX, _genvector operandY, int vectorLength,
                          cudaStream_t stream) {
#ifndef DISABLE_CUDA
  int gridSize = (vectorLength + blockSize - 1) / blockSize;
  saxpy_kernel<<<gridSize, blockSize, 0, stream>>>(result, alpha, operandX, operandY, vectorLength);
#else
  Complex *result_ = (Complex *)result;
  Complex *operandX_ = (Complex *)operandX;
  Complex *operandY_ = (Complex *)operandY;
  for (int i = 0; i < vectorLength; i++) {
    result_[i] = alpha * operandX_[i] + operandY_[i];
  }
#endif
}

// result = alpha * operand
void QcuSax::operator()(_genvector result, Complex alpha, _genvector operandX, int vectorLength, cudaStream_t stream) {
  //  void sax_kernel(void *result, Complex scalar, void *operandX, int vectorLength)
  int gridSize = (vectorLength + blockSize - 1) / blockSize;
  sax_kernel<<<gridSize, blockSize, 0, stream>>>(result, alpha, operandX, vectorLength);
}

// result = operand
void QcuComplexCopy::operator()(_genvector result, _genvector operand, int vectorLength, cudaStream_t stream) {
  int gridSize = (vectorLength + blockSize - 1) / blockSize;
  copyComplex<<<gridSize, blockSize, 0, stream>>>(result, operand, vectorLength);
}

void complexDivideGPU(void *res, void *a, void *b, cudaStream_t stream) {
  complexDivideKernel<<<1, 1, 0, stream>>>(res, a, b);
}
#endif
END_NAMESPACE(qcu)