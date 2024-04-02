#include "algebra/qcu_algebra.h"
#include "basic_data/qcu_complex.cuh"
#include "comm/qcu_communicator.h"
#include "qcu_macro.cuh"
#include <cmath>
#include <mpi.h>

template <typename T> void reduceSum(T *result, T *data, int size) {
  T sum = 0;
  for (int i = 0; i < size; i++) {
    sum += data[i];
  }
  *result = sum;
}

template <typename T> void init(T *data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = i + 1;
  }
  printf("initial function %s over\n", __FUNCTION__);
}
template <> void init<Complex>(Complex *data, int size) {
  for (size_t i = 0; i < size; i++) {
    // data[i] = Complex(i, 2 * i + 1);
    data[i] = Complex(1, 0);
  }
  printf("complex initial function %s over\n", __FUNCTION__);
}

template <typename T> double norm2(T *data, int size) {
  printf("%s donnot implement\n", __FUNCTION__);
  return 0;
}
template <> double norm2<Complex>(Complex *data, int size) {
//   printf("### COMPLEX norm2\n");
  double sum = 0;
  for (int i = 0; i < size; i++) {
    sum += data[i].norm2Square();
  }
  return sqrt(sum);
}
template <> double norm2<double>(double *data, int size) {
  double sum = 0;
  for (int i = 0; i < size; i++) {
    sum += data[i] * data[i];
  }
  return sqrt(sum);
}

void checkNorm2(void *h_data, void *d_data, void *d_resbuf, int latticeSize) {

  qcu::MsgHandler msgHandler;
  qcu::QcuNorm2 deviceNorm(&msgHandler);
  double h_res;
  double d_res;
  double *d_res_ptr;

  cudaMalloc(&d_res_ptr, sizeof(double));
  h_res = norm2((Complex *)h_data, latticeSize * 12);
  deviceNorm(static_cast<void *>(d_res_ptr), d_resbuf, d_data, latticeSize * 12);
  cudaDeviceSynchronize();
  cudaMemcpy(&d_res, d_res_ptr, sizeof(double), cudaMemcpyDeviceToHost);
  printf("h_res = %lf, d_res = %lf, ground truth = %lf\n", h_res, d_res, (h_res - d_res) / h_res);

  cudaFree(d_res_ptr);
}

// template <typename T> void checkKernel() {
int main() {
  int myRank;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  using T = Complex;
  int size = 8 * 8 * 16 * 16;
//   int size = 1 * 1 * 1 * 2;
  T *h_src;  // host src
  T *d_src;  // device src
  T *d_temp; // device result

  h_src = new T[size * 12];
  cudaMalloc(&d_src, size * 12 * sizeof(T));
  cudaMalloc(&d_temp, size * 12 * sizeof(T));

  init<T>(h_src, size * 12);
  cudaMemcpy(d_src, h_src, size * 12 * sizeof(T), cudaMemcpyHostToDevice);

  checkNorm2(h_src, d_src, d_temp, size);
  delete[] h_src;
  cudaFree(d_src);
  cudaFree(d_temp);
  MPI_Finalize();
  return 0;
}