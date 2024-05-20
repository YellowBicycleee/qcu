#include <mpi.h>
#include <pthread.h>

#include <thread>

#include "qcd/qcu_wilson_dslash.cuh"
#include "targets/wilson_dslash_ghost_kernel.cuh"
#include "targets/wilson_dslash_kernel.cuh"

static std::thread threads[Nd * DIRECTIONS];

#ifdef DEBUG
char dslashDimName[4][6] = {"X_DIM", "Y_DIM", "Z_DIM", "T_DIM"};
char dslashDirName[2][4] = {"BWD", "FWD"};
#endif

BEGIN_NAMESPACE(qcu)

// use this function to call kernel function, this function donnot sync inside
// void WilsonDslash::apply(int daggerFlag) {
void WilsonDslash::apply() {
  int daggerFlag = dslashParam_->daggerFlag;
  void *gauge = dslashParam_->gauge;
  void *fermionIn = dslashParam_->fermionIn;
  void *fermionOut = dslashParam_->fermionOut;
  int parity = dslashParam_->parity;
  int Lx = dslashParam_->Lx;
  int Ly = dslashParam_->Ly;
  int Lz = dslashParam_->Lz;
  int Lt = dslashParam_->Lt;
  int Nx = dslashParam_->Nx;
  int Ny = dslashParam_->Ny;
  int Nz = dslashParam_->Nz;
  int Nt = dslashParam_->Nt;

  cudaStream_t stream1 = dslashParam_->stream1;
  double daggerParam = daggerFlag ? -1.0 : 1.0;
  int vol = Lx * Ly * Lz * Lt;

  int gridSize = (vol / 2 + blockSize_ - 1) / blockSize_;
  dslashKernelFunc<<<gridSize, blockSize_, 0, stream1>>>(gauge, fermionIn, fermionOut, Lx, Ly, Lz, Lt, parity, Nx, Ny,
                                                         Nz, Nt, daggerParam);
}

void WilsonDslash::postDslashMPI(int dim, int dir, int daggerFlag) {
  if ((dim < X_DIM || dim > T_DIM) || (dir < 0 || dir > 1)) {
    return;
  }
  dslashMPIWait(dim, dir);

#ifdef DEBUG
  printf("in function %s, line %d, dim = %s, dir = %s, pos = %d\n", __FUNCTION__, __LINE__, dslashDimName[dim],
         dslashDirName[dir], dim * DIRECTIONS + dir);

#endif
  // cudaStream_t stream = dslashParam_->commStreams[dim * DIRECTIONS + dir];
  cudaStream_t stream = dslashParam_->stream1; 
  void *gauge = dslashParam_->gauge;
  void *fermionIn = dslashParam_->memPool->d_recv_buffer[dim][dir];
  void *fermionOut = dslashParam_->fermionOut;
  int parity = dslashParam_->parity;

  double daggerParam = daggerFlag ? -1.0 : 1.0;
  int vol = 1;
  int lattSize[Nd] = {dslashParam_->Lx, dslashParam_->Ly, dslashParam_->Lz, dslashParam_->Lt};
  vol = lattSize[X_DIM] * lattSize[Y_DIM] * lattSize[Z_DIM] * lattSize[T_DIM] / 2 / lattSize[dim];

  int gridSize = (vol + blockSize_ - 1) / blockSize_;
  if (dim == X_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryX<<<gridSize, blockSize_, 0, stream>>>(fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                  lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                  static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryX<<<gridSize, blockSize_, 0, stream>>>(gauge, fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                   lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                   static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == Y_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryY<<<gridSize, blockSize_, 0, stream>>>(fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                  lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                  static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryY<<<gridSize, blockSize_, 0, stream>>>(gauge, fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                   lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                   static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == Z_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryZ<<<gridSize, blockSize_, 0, stream>>>(fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                  lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                  static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryZ<<<gridSize, blockSize_, 0, stream>>>(gauge, fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                   lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                   static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == T_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryT<<<gridSize, blockSize_, 0, stream>>>(fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                  lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                  static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryT<<<gridSize, blockSize_, 0, stream>>>(gauge, fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                   lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                   static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else {
    assert(0);
  }
  // CHECK_CUDA(cudaStreamSynchronize(stream));
}

void WilsonDslash::preDslashMPI(int dim, int dir, int daggerFlag) {
  if ((dim < X_DIM || dim > T_DIM) || (dir < 0 || dir > 1)) {
    return;
  }
  cudaStream_t stream = dslashParam_->commStreams[dim * DIRECTIONS + dir];
  void *gauge = dslashParam_->gauge;
  void *fermionIn = dslashParam_->fermionIn;
  void *fermionOut = dslashParam_->memPool->d_send_buffer[dim][dir];
  int parity = dslashParam_->parity;
  int Lx = dslashParam_->Lx;
  int Ly = dslashParam_->Ly;
  int Lz = dslashParam_->Lz;
  int Lt = dslashParam_->Lt;
  int Nx = dslashParam_->Nx;
  int Ny = dslashParam_->Ny;
  int Nz = dslashParam_->Nz;
  int Nt = dslashParam_->Nt;

  void *h_fermionOut = dslashParam_->memPool->h_send_buffer[dim][dir];

  double daggerParam = daggerFlag ? -1.0 : 1.0;
  int vol = 1;
  int temp[Nd] = {Lx, Ly, Lz, Lt};
  temp[dim] = 1;
  for (int i = 0; i < Nd; i++) {
    vol *= temp[i];
  }
  vol /= 2;

  int gridSize = (vol + blockSize_ - 1) / blockSize_;
  if (dim == X_DIM) {
    if (dir == BWD) {
      DslashTransferBackX<<<gridSize, blockSize_, 0, stream>>>(fermionIn, Lx, Ly, Lz, Lt, parity,
                                                               static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontX<<<gridSize, blockSize_, 0, stream>>>(gauge, fermionIn, Lx, Ly, Lz, Lt, parity,
                                                                static_cast<Complex *>(fermionOut), daggerParam);
    }
  } else if (dim == Y_DIM) {
    if (dir == BWD) {
      DslashTransferBackY<<<gridSize, blockSize_, 0, stream>>>(fermionIn, Lx, Ly, Lz, Lt, parity,
                                                               static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontY<<<gridSize, blockSize_, 0, stream>>>(gauge, fermionIn, Lx, Ly, Lz, Lt, parity,
                                                                static_cast<Complex *>(fermionOut), daggerParam);
    }
  } else if (dim == Z_DIM) {
    if (dir == BWD) {
      DslashTransferBackZ<<<gridSize, blockSize_, 0, stream>>>(fermionIn, Lx, Ly, Lz, Lt, parity,
                                                               static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontZ<<<gridSize, blockSize_, 0, stream>>>(gauge, fermionIn, Lx, Ly, Lz, Lt, parity,
                                                                static_cast<Complex *>(fermionOut), daggerParam);
    }
  } else if (dim == T_DIM) {
    if (dir == BWD) {
      DslashTransferBackT<<<gridSize, blockSize_, 0, stream>>>(fermionIn, Lx, Ly, Lz, Lt, parity,
                                                               static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontT<<<gridSize, blockSize_, 0, stream>>>(gauge, fermionIn, Lx, Ly, Lz, Lt, parity,
                                                                static_cast<Complex *>(fermionOut), daggerParam);
    }
  } else {
    return;
  }
  CHECK_CUDA(
      cudaMemcpyAsync(h_fermionOut, fermionOut, vol * Ns * Nc * 2 * sizeof(double), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  MPI_Start(&(dslashParam_->msgHandler->mpiSendRequest[dim][dir]));
  MPI_Start(&(dslashParam_->msgHandler->mpiRecvRequest[dim][dir]));
}
// void WilsonDslash::dslashMPIIsendrecv(int dim) {
//   MPI_Start(&(dslashParam_->msgHandler->mpiSendRequest[dim][FWD]));
//   MPI_Start(&(dslashParam_->msgHandler->mpiSendRequest[dim][BWD]));
//   MPI_Start(&(dslashParam_->msgHandler->mpiRecvRequest[dim][FWD]));
//   MPI_Start(&(dslashParam_->msgHandler->mpiRecvRequest[dim][BWD]));
// }

void WilsonDslash::dslashMPIWait(int dim, int dir) {
  int sendLength;
  int lattSize[4] = {dslashParam_->Lx, dslashParam_->Ly, dslashParam_->Lz, dslashParam_->Lt};
  sendLength = lattSize[X_DIM] * lattSize[Y_DIM] * lattSize[Z_DIM] * lattSize[T_DIM] / 2 / lattSize[dim] * Ns * Nc;
  cudaStream_t stream = dslashParam_->stream1;
  switch (dir) {
    case FWD:
      // stream = dslashParam_->commStreams[dim * DIRECTIONS + FWD];
      CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[dim][FWD], MPI_STATUS_IGNORE));
      CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiSendRequest[dim][FWD], MPI_STATUS_IGNORE));
      CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[dim][FWD],
                                 dslashParam_->memPool->h_recv_buffer[dim][FWD], sendLength * 2 * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      break;
    case BWD:
      // stream = dslashParam_->commStreams[dim * DIRECTIONS + BWD];
      CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[dim][BWD], MPI_STATUS_IGNORE));
      CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiSendRequest[dim][BWD], MPI_STATUS_IGNORE));
      CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[dim][BWD],
                                 dslashParam_->memPool->h_recv_buffer[dim][BWD], sendLength * 2 * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      break;
    default:
      assert(0);
      break;
  }
}

void WilsonDslash::cudaStreamBarrier() {
  cudaStream_t streamFwd;
  cudaStream_t streamBwd;
  int mpiGridDim[Nd] = {dslashParam_->Nx, dslashParam_->Ny, dslashParam_->Nz, dslashParam_->Nt};
  for (int i = 0; i < Nd; i++) {
    if (mpiGridDim[i] > 1) {
      streamFwd = dslashParam_->commStreams[i * DIRECTIONS + FWD];
      streamBwd = dslashParam_->commStreams[i * DIRECTIONS + BWD];
      CHECK_CUDA(cudaStreamSynchronize(streamFwd));
      CHECK_CUDA(cudaStreamSynchronize(streamBwd));
    }
  }
}

void WilsonDslash::preApply2() {
  int daggerFlag = dslashParam_->daggerFlag;

  int mpiGridDim[Nd] = {dslashParam_->Nx, dslashParam_->Ny, dslashParam_->Nz, dslashParam_->Nt};
  // Parameters parameters[Nd * DIRECTIONS];
#pragma unroll
  for (int dim = X_DIM; dim < Nd; dim++) {
    if (mpiGridDim[dim] > 1) {
      // preDslashMPI(dim, FWD, daggerFlag);
      //  preDslashMPI(dim, BWD, daggerFlag);
      for (int dir = 0; dir < 2; dir++) {
        threads[dim * DIRECTIONS + dir] = std::thread(&WilsonDslash::preDslashMPI, this, dim, dir, daggerFlag);
      }
    }
  }
  // BARRIER
  // cudaStreamBarrier();

  // #pragma unroll
  //   // SendRecv
  //   for (int dim = X_DIM; dim < Nd; dim++) {
  //     if (mpiGridDim[dim] > 1) {
  //       dslashMPIIsendrecv(dim);
  //     }
  //   }
}

void WilsonDslash::postApply2() {
  // WAIT
  CHECK_CUDA(cudaStreamSynchronize(dslashParam_->stream1));
  int mpiGridDim[Nd] = {dslashParam_->Nx, dslashParam_->Ny, dslashParam_->Nz, dslashParam_->Nt};
  int daggerFlag = dslashParam_->daggerFlag;

  // cudaStreamBarrier();

#pragma unroll
  // calculate
  for (int dim = X_DIM; dim < Nd; dim++) {
    if (mpiGridDim[dim] > 1) {
      for (int i = 0; i < DIRECTIONS; i++) {
        threads[dim * DIRECTIONS + i].join();
        threads[dim * DIRECTIONS + i] = std::thread(&WilsonDslash::postDslashMPI, this, dim, i, daggerFlag);
      }
    }
  }
  for (int dim = X_DIM; dim < Nd; dim++) {
    if (mpiGridDim[dim] > 1) {
      for (int i = 0; i < DIRECTIONS; i++) {
        threads[dim * DIRECTIONS + i].join();
      }
    }
  }
  CHECK_CUDA(cudaStreamSynchronize(dslashParam_->stream1));
  // cudaStreamBarrier();
  // CHECK_CUDA(cudaStreamSynchronize(dslashParam_->stream1));
}

END_NAMESPACE(qcu)