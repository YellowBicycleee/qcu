#include <mpi.h>

#include "qcd/qcu_wilson_dslash.cuh"
#include "targets/wilson_dslash_ghost_kernel.cuh"
#include "targets/wilson_dslash_kernel.cuh"

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

// only call kernel funcs
void WilsonDslash::preDslash(int dim, int dir, int daggerFlag) {
  if ((dim < X_DIM || dim > T_DIM) || (dir < 0 || dir > 1)) {
    return;
  }
  cudaStream_t stream2 = dslashParam_->stream2;
  void *gauge = dslashParam_->gauge;
  void *fermionIn = dslashParam_->fermionIn;
  void *fermionOut = dslashParam_->memPool->d_send_buffer[dim][dir];

  int parity = dslashParam_->parity;
  int lattSize[Nd] = {dslashParam_->Lx, dslashParam_->Ly, dslashParam_->Lz, dslashParam_->Lt};
  int vol = lattSize[X_DIM] * lattSize[Y_DIM] * lattSize[Z_DIM] * lattSize[T_DIM] / 2 / lattSize[dim];

  double daggerParam = daggerFlag ? -1.0 : 1.0;

  int gridSize = (vol + blockSize_ - 1) / blockSize_;
  if (dim == X_DIM) {
    if (dir == BWD) {
      DslashTransferBackX<<<gridSize, blockSize_, 0, stream2>>>(fermionIn, lattSize[X_DIM], lattSize[Y_DIM],
                                                                lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontX<<<gridSize, blockSize_, 0, stream2>>>(gauge, fermionIn, lattSize[X_DIM], lattSize[Y_DIM],
                                                                 lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                 static_cast<Complex *>(fermionOut), daggerParam);
    }
  } else if (dim == Y_DIM) {
    if (dir == BWD) {
      DslashTransferBackY<<<gridSize, blockSize_, 0, stream2>>>(fermionIn, lattSize[X_DIM], lattSize[Y_DIM],
                                                                lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontY<<<gridSize, blockSize_, 0, stream2>>>(gauge, fermionIn, lattSize[X_DIM], lattSize[Y_DIM],
                                                                 lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                 static_cast<Complex *>(fermionOut), daggerParam);
    }
  } else if (dim == Z_DIM) {
    if (dir == BWD) {
      DslashTransferBackZ<<<gridSize, blockSize_, 0, stream2>>>(fermionIn, lattSize[X_DIM], lattSize[Y_DIM],
                                                                lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontZ<<<gridSize, blockSize_, 0, stream2>>>(gauge, fermionIn, lattSize[X_DIM], lattSize[Y_DIM],
                                                                 lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                 static_cast<Complex *>(fermionOut), daggerParam);
    }
  } else if (dim == T_DIM) {
    if (dir == BWD) {
      DslashTransferBackT<<<gridSize, blockSize_, 0, stream2>>>(fermionIn, lattSize[X_DIM], lattSize[Y_DIM],
                                                                lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontT<<<gridSize, blockSize_, 0, stream2>>>(gauge, fermionIn, lattSize[X_DIM], lattSize[Y_DIM],
                                                                 lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                 static_cast<Complex *>(fermionOut), daggerParam);
    }
  } else {
    assert(0);
  }
}

void WilsonDslash::postDslash(int dim, int dir, int daggerFlag) {
  if ((dim < X_DIM || dim > T_DIM) || (dir < 0 || dir > 1)) {
    return;
  }
  cudaStream_t stream1 = dslashParam_->stream1;
  void *gauge = dslashParam_->gauge;
  void *fermionIn = dslashParam_->memPool->d_recv_buffer[dim][dir];
  void *fermionOut = dslashParam_->fermionOut;
  int parity = dslashParam_->parity;
  int lattSize[Nd] = {dslashParam_->Lx, dslashParam_->Ly, dslashParam_->Lz, dslashParam_->Lt};
  int vol = lattSize[X_DIM] * lattSize[Y_DIM] * lattSize[Z_DIM] * lattSize[T_DIM] / 2 / lattSize[dim];
  double daggerParam = daggerFlag ? -1.0 : 1.0;

  int gridSize = (vol + blockSize_ - 1) / blockSize_;
  if (dim == X_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryX<<<gridSize, blockSize_, 0, stream1>>>(fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                   lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                   static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryX<<<gridSize, blockSize_, 0, stream1>>>(gauge, fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                    lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                    static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == Y_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryY<<<gridSize, blockSize_, 0, stream1>>>(fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                   lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                   static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryY<<<gridSize, blockSize_, 0, stream1>>>(gauge, fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                    lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                    static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == Z_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryZ<<<gridSize, blockSize_, 0, stream1>>>(fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                   lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                   static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryZ<<<gridSize, blockSize_, 0, stream1>>>(gauge, fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                    lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                    static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == T_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryT<<<gridSize, blockSize_, 0, stream1>>>(fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                   lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                   static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryT<<<gridSize, blockSize_, 0, stream1>>>(gauge, fermionOut, lattSize[X_DIM], lattSize[Y_DIM],
                                                                    lattSize[Z_DIM], lattSize[T_DIM], parity,
                                                                    static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else {
    assert(0);
  }
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
  cudaStream_t stream = dslashParam_->commStreams[dim * DIRECTIONS + dir];
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

void WilsonDslash::dslashNcclIsendrecv(int dim) {
  void *sendbuf;
  void *recvbuf;
  int dest, src;
  cudaStream_t stream2 = dslashParam_->stream2;
  int lattSize[Nd] = {dslashParam_->Lx, dslashParam_->Ly, dslashParam_->Lz, dslashParam_->Lt};
  int vectorLength =
      lattSize[X_DIM] * lattSize[Y_DIM] * lattSize[Z_DIM] * lattSize[T_DIM] / 2 / lattSize[dim] * Ns * Nc;

  ncclComm_t &comm = dslashParam_->msgHandler->ncclComm;
  // CHECK_NCCL(ncclGroupStart());
  // send to fwd, and send from bwd
  sendbuf = dslashParam_->memPool->d_send_buffer[dim][FWD];
  recvbuf = dslashParam_->memPool->d_recv_buffer[dim][BWD];
  dest = dslashParam_->qcuComm->getNeighborRank(dim, FWD);
  src = dslashParam_->qcuComm->getNeighborRank(dim, BWD);

  // CHECK_NCCL(ncclGroupStart());
  CHECK_NCCL(ncclSend(sendbuf, vectorLength * 2, ncclDouble, dest, comm, stream2));
  CHECK_NCCL(ncclRecv(recvbuf, vectorLength * 2, ncclDouble, src, comm, stream2));
  // CHECK_NCCL(ncclGroupEnd());

  // CHECK_CUDA(cudaStreamSynchronize(cudaStream2_));
  // send to bwd, and send from fwd
  sendbuf = dslashParam_->memPool->d_send_buffer[dim][BWD];
  recvbuf = dslashParam_->memPool->d_recv_buffer[dim][FWD];
  dest = dslashParam_->qcuComm->getNeighborRank(dim, BWD);
  src = dslashParam_->qcuComm->getNeighborRank(dim, FWD);
  // CHECK_NCCL(ncclGroupStart());
  CHECK_NCCL(ncclSend(sendbuf, vectorLength * 2, ncclDouble, dest, comm, stream2));
  CHECK_NCCL(ncclRecv(recvbuf, vectorLength * 2, ncclDouble, src, comm, stream2));
  // CHECK_NCCL(ncclGroupEnd());
}

void WilsonDslash::dslashNcclWait() {
  cudaStream_t stream2 = dslashParam_->stream2;
  CHECK_CUDA(cudaStreamSynchronize(stream2));
}

void WilsonDslash::preApply() {
  int daggerFlag = dslashParam_->daggerFlag;
  int mpiGridDim[Nd] = {dslashParam_->Nx, dslashParam_->Ny, dslashParam_->Nz, dslashParam_->Nt};
  // boundary calc and transfer on stream2
  for (int i = 0; i < Nd; i++) {
    if (mpiGridDim[i] > 1) {
      preDslash(i, FWD, daggerFlag);
      preDslash(i, BWD, daggerFlag);
    }
  }
  // if (dslashParam_->Nx > 1) {
  //   preDslash(X_DIM, FWD, daggerFlag);
  //   preDslash(X_DIM, BWD, daggerFlag);
  // }
  // if (dslashParam_->Ny > 1) {
  //   preDslash(Y_DIM, FWD, daggerFlag);
  //   preDslash(Y_DIM, BWD, daggerFlag);
  // }
  // if (dslashParam_->Nz > 1) {
  //   preDslash(Z_DIM, FWD, daggerFlag);
  //   preDslash(Z_DIM, BWD, daggerFlag);
  // }
  // if (dslashParam_->Nt > 1) {
  //   preDslash(T_DIM, FWD, daggerFlag);
  //   preDslash(T_DIM, BWD, daggerFlag);
  // }
  // SYNC GPU
  // CHECK_CUDA(cudaStreamSynchronize(cudaStream1_));
  // CHECK_CUDA(cudaStreamSynchronize(cudaStream2_));
  // MPI_Isend MPI_Irecv
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < Nd; i++) {
    if (mpiGridDim[i] > 1) {
      dslashNcclIsendrecv(i);
    }
  }
  // if (dslashParam_->Nx > 1) {
  //   dslashNcclIsendrecv(X_DIM);
  // }

  // if (dslashParam_->Ny > 1) {
  //   dslashNcclIsendrecv(Y_DIM);
  // }

  // if (dslashParam_->Nz > 1) {
  //   dslashNcclIsendrecv(Z_DIM);
  // }

  // if (dslashParam_->Nt > 1) {
  //   dslashNcclIsendrecv(T_DIM);
  // }
  CHECK_NCCL(ncclGroupEnd());
  // #ifdef DEBUG
  //   printf("PreApply Done\n");
  // #endif
}

void WilsonDslash::postApply() {
  int daggerFlag = dslashParam_->daggerFlag;
  // MPI_Wait
  if (dslashParam_->Nx > 1 || dslashParam_->Ny > 1 || dslashParam_->Nz > 1 || dslashParam_->Nt > 1) {
    dslashNcclWait();
  }

  // calculate
  if (dslashParam_->Nx > 1) {
    postDslash(X_DIM, FWD, daggerFlag);
    postDslash(X_DIM, BWD, daggerFlag);
  }
  if (dslashParam_->Ny > 1) {
    postDslash(Y_DIM, FWD, daggerFlag);
    postDslash(Y_DIM, BWD, daggerFlag);
  }
  if (dslashParam_->Nz > 1) {
    postDslash(Z_DIM, FWD, daggerFlag);
    postDslash(Z_DIM, BWD, daggerFlag);
  }
  if (dslashParam_->Nt > 1) {
    postDslash(T_DIM, FWD, daggerFlag);
    postDslash(T_DIM, BWD, daggerFlag);
  }
  // printf("PostApply Done\n");
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
}
void WilsonDslash::dslashMPIIsendrecv(int dim) {
#ifdef MPI_START_SENDRECV
  MPI_Start(&(dslashParam_->msgHandler->mpiSendRequest[dim][FWD]));
  MPI_Start(&(dslashParam_->msgHandler->mpiSendRequest[dim][BWD]));
  MPI_Start(&(dslashParam_->msgHandler->mpiRecvRequest[dim][FWD]));
  MPI_Start(&(dslashParam_->msgHandler->mpiRecvRequest[dim][BWD]));
#else
  int lattSize[Nd] = {dslashParam_->Lx, dslashParam_->Ly, dslashParam_->Lz, dslashParam_->Lt};
  int sendLength = lattSize[X_DIM] * lattSize[Y_DIM] * lattSize[Z_DIM] * lattSize[T_DIM] / 2 / lattSize[dim] * Ns * Nc;
  // for (int i = 0; i < Nd; i++) {
  {
    cudaStream_t streamFwd = dslashParam_->commStreams[dim * DIRECTIONS + FWD];
    cudaStream_t streamBwd = dslashParam_->commStreams[dim * DIRECTIONS + BWD];
    void *sendbuf;
    void *recvbuf;
    int dest, src;

    // FWD:
    sendbuf = dslashParam_->memPool->h_send_buffer[dim][FWD];
    recvbuf = dslashParam_->memPool->h_recv_buffer[dim][BWD];
    dest = dslashParam_->qcuComm->getNeighborRank(dim, FWD);
    src = dslashParam_->qcuComm->getNeighborRank(dim, BWD);
    // to fwd, tag = FWD
    CHECK_MPI(MPI_Isend(sendbuf, sendLength * 2, MPI_DOUBLE, dest, FWD, MPI_COMM_WORLD,
                        &dslashParam_->msgHandler->mpiSendRequest[dim][FWD]));
    // from bwd, when bwd send msg, tag is FWD
    CHECK_MPI(MPI_Irecv(recvbuf, sendLength * 2, MPI_DOUBLE, src, FWD, MPI_COMM_WORLD,
                        &dslashParam_->msgHandler->mpiRecvRequest[dim][BWD]));
    // BWD:
    sendbuf = dslashParam_->memPool->h_send_buffer[dim][BWD];
    recvbuf = dslashParam_->memPool->h_recv_buffer[dim][FWD];
    dest = dslashParam_->qcuComm->getNeighborRank(dim, BWD);
    src = dslashParam_->qcuComm->getNeighborRank(dim, FWD);
    // to bwd, tag = BWD
    CHECK_MPI(MPI_Isend(sendbuf, sendLength * 2, MPI_DOUBLE, dest, BWD, MPI_COMM_WORLD,
                        &dslashParam_->msgHandler->mpiSendRequest[dim][BWD]));
    // from bwd, when bwd send msg, tag is fwd
    CHECK_MPI(MPI_Irecv(recvbuf, sendLength * 2, MPI_DOUBLE, src, BWD, MPI_COMM_WORLD,
                        &dslashParam_->msgHandler->mpiRecvRequest[dim][FWD]));
  }
#endif
}

void WilsonDslash::dslashMPIWait(int dim, int dir) {
  int sendLength;
  int lattSize[4] = {dslashParam_->Lx, dslashParam_->Ly, dslashParam_->Lz, dslashParam_->Lt};
  sendLength = lattSize[X_DIM] * lattSize[Y_DIM] * lattSize[Z_DIM] * lattSize[T_DIM] / 2 / lattSize[dim] * Ns * Nc;
  cudaStream_t stream;
  switch (dir) {
    case FWD:
      stream = dslashParam_->commStreams[dim * DIRECTIONS + FWD];
      CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[dim][FWD], MPI_STATUS_IGNORE));
      CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiSendRequest[dim][FWD], MPI_STATUS_IGNORE));
      CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[dim][FWD],
                                 dslashParam_->memPool->h_recv_buffer[dim][FWD], sendLength * 2 * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      break;
    case BWD:
      stream = dslashParam_->commStreams[dim * DIRECTIONS + BWD];
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

#pragma unroll
  for (int dim = X_DIM; dim < Nd; dim++) {
    if (mpiGridDim[dim] > 1) {
      preDslashMPI(dim, FWD, daggerFlag);
      preDslashMPI(dim, BWD, daggerFlag);
    }
  }

  // BARRIER
  cudaStreamBarrier();

#pragma unroll
  // SendRecv
  for (int dim = X_DIM; dim < Nd; dim++) {
    if (mpiGridDim[dim] > 1) {
      dslashMPIIsendrecv(dim);
    }
  }
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
      postDslashMPI(dim, FWD, daggerFlag);  // D2H + kernel
      postDslashMPI(dim, BWD, daggerFlag);
      CHECK_CUDA(cudaStreamSynchronize(dslashParam_->commStreams[dim * DIRECTIONS + FWD]));
      CHECK_CUDA(cudaStreamSynchronize(dslashParam_->commStreams[dim * DIRECTIONS + BWD]));
    }
  }
  cudaStreamBarrier();
  // CHECK_CUDA(cudaStreamSynchronize(dslashParam_->stream1));
}

END_NAMESPACE(qcu)