#pragma once

#include "basic_data/qcu_complex.cuh"
__global__ void complexDivideKernel(void *result, void *a, void *b) {
  Complex x = *static_cast<Complex *>(a);
  Complex y = *static_cast<Complex *>(b);
  Complex ret = x / y;
  *static_cast<Complex *>(result) = ret;
}

__global__ void doubleSqrt(void *result, void *operand) {
  double x = *static_cast<double *>(operand);
  double ret = sqrt(x);
  *static_cast<double *>(result) = ret;
}

__global__ void double2VectorAdd(void *result, void *operand1, void *operand2, int vectorLength) {
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  int totalSize = blockDim.x * gridDim.x;
  double2 x_in;
  double2 y_in;
  double x1, x2;
  double y1, y2;

  for (int i = globalId; i < vectorLength; i += totalSize) {
    // static_cast<double *>(result)[i] = static_cast<double *>(operand1)[i] + static_cast<double
    // *>(operand2)[i];
    x_in = static_cast<double2 *>(operand1)[i];
    y_in = static_cast<double2 *>(operand2)[i];
    x1 = x_in.x;
    x2 = x_in.y;
    y1 = y_in.x;
    y2 = y_in.y;
    static_cast<double2 *>(result)[i] = make_double2(x1 + y1, x2 + y2);
  }
}

__global__ void complexReduceSum(void *result, void *src, int vectorLength) {
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  int localId = threadIdx.x;
  int totalSize = blockDim.x * gridDim.x;
  int blockSize = blockDim.x;

  // set length by the third parameter of the kernel launch
  extern __shared__ Complex complexSharedMemory[];
  complexSharedMemory[localId].clear2Zero();
  __syncthreads();

  for (int i = globalId; i < vectorLength; i += totalSize) {
    complexSharedMemory[localId] += static_cast<Complex *>(src)[i];
  }
  __syncthreads();

  // reduce the shared memory
  for (int stride = blockSize / 2; stride > 0; stride >>= 1) { // i represents the stride
    if (localId < stride) {
      complexSharedMemory[localId] += complexSharedMemory[localId + stride];
    }
    __syncthreads();
  }
  // result is stored in complexSharedMemory[0]
  if (localId == 0) {
    static_cast<Complex *>(result)[blockIdx.x] = complexSharedMemory[0];
  }
}

__global__ void doubleReduceSum(void *result, void *src, int vectorLength) {
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  int localId = threadIdx.x;
  int totalSize = blockDim.x * gridDim.x;
  int blockSize = blockDim.x;

  // set length by the third parameter of the kernel launch
  extern __shared__ double doubleSharedMemory[];
  doubleSharedMemory[localId] = 0;
  __syncthreads();

  for (int i = globalId; i < vectorLength; i += totalSize) {
    doubleSharedMemory[localId] += static_cast<double *>(src)[i];
  }
  __syncthreads();

  // reduce the shared memory
  for (int stride = blockSize / 2; stride > 0; stride >>= 1) { // i represents the stride
    if (localId < stride) {
      doubleSharedMemory[localId] += doubleSharedMemory[localId + stride];
    }
    __syncthreads();
  }
  // result is stored in doubleSharedMemory[0]
  if (localId == 0) {
    static_cast<double *>(result)[blockIdx.x] = doubleSharedMemory[0];
  }
}

__global__ void innerProduct(void *result, void *operand1, void *operand2, int vectorLength) {
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  int localId = threadIdx.x;
  int totalSize = blockDim.x * gridDim.x;
  int blockSize = blockDim.x;

  // set length by the third parameter of the kernel launch
  extern __shared__ Complex complexSharedMemory[];
  Complex *x;
  Complex *y;

  complexSharedMemory[localId].clear2Zero();
  __syncthreads();

  for (int i = globalId; i < vectorLength; i += totalSize) {
    x = reinterpret_cast<Complex *>(operand1) + i;
    y = reinterpret_cast<Complex *>(operand2) + i;
    complexSharedMemory[localId] += (*x) * (*y).conj();
  }
  __syncthreads();

  // reduce the shared memory
  for (int stride = blockSize / 2; stride > 0; stride >>= 1) { // i represents the stride
    if (localId < stride) {
      complexSharedMemory[localId] += complexSharedMemory[localId + stride];
    }
    __syncthreads();
  }
  // result is stored in sharedMemory[0]
  if (localId == 0) {
    static_cast<Complex *>(result)[blockIdx.x] = complexSharedMemory[0];
  }
}

// single process norm2Square
__global__ void norm2Square(void *result, void *operand, int vectorLength) {
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  int localId = threadIdx.x;
  int totalSize = blockDim.x * gridDim.x;
  int blockSize = blockDim.x;

  // set length by the third parameter of the kernel launch
  extern __shared__ double doubleSharedMemory[];
  double2 *x_in;

  doubleSharedMemory[localId] = 0;
  __syncthreads();

  for (int i = globalId; i < vectorLength; i += totalSize) {
    x_in = reinterpret_cast<double2 *>(operand) + i;
    doubleSharedMemory[localId] += Complex(*x_in).norm2Square();
  }

  __syncthreads();

  // reduce the shared memory
  for (int stride = blockSize / 2; stride > 0; stride >>= 1) { // i represents the stride
    if (localId < stride) {
      doubleSharedMemory[localId] += doubleSharedMemory[localId + stride];
    }

    __syncthreads();
  }

  // result is stored in doubleSharedMemory[0]
  if (localId == 0) {
    static_cast<double *>(result)[blockIdx.x] = doubleSharedMemory[0];
  }
}

// ---------------------------------------
// simple linear algebra
/// Saxpy
__global__ void saxpy_kernel(void *result, Complex scalar, void *operandX, void *operandY,
                             int vectorLength) {
  double2 *resultPtr;
  double2 *operandXPtr;
  double2 *operandYPtr;
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  int totalSize = blockDim.x * gridDim.x;
  Complex temp;

  for (int i = globalId; i < vectorLength; i += totalSize) {
    resultPtr = static_cast<double2 *>(result) + i;
    operandXPtr = static_cast<double2 *>(operandX) + i;
    operandYPtr = static_cast<double2 *>(operandY) + i;
    temp = scalar * Complex(*operandXPtr) + Complex(*operandYPtr);
    *resultPtr = make_double2(temp.real(), temp.imag());
  }
}

/// Sax
__global__ void sax_kernel(void *result, Complex scalar, void *operandX, int vectorLength) {
  double2 *resultPtr;
  double2 *operandXPtr;
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  int totalSize = blockDim.x * gridDim.x;
  Complex temp;

  for (int i = globalId; i < vectorLength; i += totalSize) {
    resultPtr = static_cast<double2 *>(result) + i;
    operandXPtr = static_cast<double2 *>(operandX) + i;
    temp = scalar * (*operandXPtr);
    *resultPtr = make_double2(temp.real(), temp.imag());
  }
}

// copy
__global__ void copyComplex(void *result, void *src, int vectorLength) {
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  int totalSize = blockDim.x * gridDim.x;

  for (int i = globalId; i < vectorLength; i += totalSize) {
    static_cast<double2 *>(result)[i] = static_cast<double2 *>(src)[i];
  }
}