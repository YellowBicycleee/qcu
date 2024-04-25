#include <mpi.h>

#include "qcd/qcu_wilson_dslash.cuh"
#include "targets/wilson_dslash_ghost_kernel.cuh"
#include "targets/wilson_dslash_kernel.cuh"
// #define DEBUG

// #define DEBUG

#ifdef DEBUG
char dslashDimName[4][6] = {"X_DIM", "Y_DIM", "Z_DIM", "T_DIM"};
char dslashDirName[2][4] = {"BWD", "FWD"};
#endif

BEGIN_NAMESPACE(qcu)

// #ifdef DEBUG
// MPI_Status status[Nd][DIRECTIONS];
// MPI_Request request[Nd][DIRECTIONS];
// #endif

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
  int Lx = dslashParam_->Lx;
  int Ly = dslashParam_->Ly;
  int Lz = dslashParam_->Lz;
  int Lt = dslashParam_->Lt;
  int Nx = dslashParam_->Nx;
  int Ny = dslashParam_->Ny;
  int Nz = dslashParam_->Nz;
  int Nt = dslashParam_->Nt;

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
      DslashTransferBackX<<<gridSize, blockSize_, 0, stream2>>>(fermionIn, Lx, Ly, Lz, Lt, parity,
                                                                static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontX<<<gridSize, blockSize_, 0, stream2>>>(gauge, fermionIn, Lx, Ly, Lz, Lt, parity,
                                                                 static_cast<Complex *>(fermionOut), daggerParam);
    }
  } else if (dim == Y_DIM) {
    if (dir == BWD) {
      DslashTransferBackY<<<gridSize, blockSize_, 0, stream2>>>(fermionIn, Lx, Ly, Lz, Lt, parity,
                                                                static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontY<<<gridSize, blockSize_, 0, stream2>>>(gauge, fermionIn, Lx, Ly, Lz, Lt, parity,
                                                                 static_cast<Complex *>(fermionOut), daggerParam);
    }
  } else if (dim == Z_DIM) {
    if (dir == BWD) {
      DslashTransferBackZ<<<gridSize, blockSize_, 0, stream2>>>(fermionIn, Lx, Ly, Lz, Lt, parity,
                                                                static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontZ<<<gridSize, blockSize_, 0, stream2>>>(gauge, fermionIn, Lx, Ly, Lz, Lt, parity,
                                                                 static_cast<Complex *>(fermionOut), daggerParam);
    }
  } else if (dim == T_DIM) {
    if (dir == BWD) {
      DslashTransferBackT<<<gridSize, blockSize_, 0, stream2>>>(fermionIn, Lx, Ly, Lz, Lt, parity,
                                                                static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontT<<<gridSize, blockSize_, 0, stream2>>>(gauge, fermionIn, Lx, Ly, Lz, Lt, parity,
                                                                 static_cast<Complex *>(fermionOut), daggerParam);
    }
  } else {
    return;
  }
}

void WilsonDslash::postDslash(int dim, int dir, int daggerFlag) {
  if ((dim < X_DIM || dim > T_DIM) || (dir < 0 || dir > 1)) {
    return;
  }
  // cudaStream_t stream1 = dslashParam_->stream1;
  cudaStream_t stream1 = dslashParam_->stream1;
  void *gauge = dslashParam_->gauge;
  // void *fermionIn = dslashParam_->fermionIn;
  void *fermionIn = dslashParam_->memPool->d_recv_buffer[dim][dir];
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
      calculateBackBoundaryX<<<gridSize, blockSize_, 0, stream1>>>(fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                   static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryX<<<gridSize, blockSize_, 0, stream1>>>(gauge, fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                    static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == Y_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryY<<<gridSize, blockSize_, 0, stream1>>>(fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                   static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryY<<<gridSize, blockSize_, 0, stream1>>>(gauge, fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                    static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == Z_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryZ<<<gridSize, blockSize_, 0, stream1>>>(fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                   static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryZ<<<gridSize, blockSize_, 0, stream1>>>(gauge, fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                    static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == T_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryT<<<gridSize, blockSize_, 0, stream1>>>(fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                   static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryT<<<gridSize, blockSize_, 0, stream1>>>(gauge, fermionOut, Lx, Ly, Lz, Lt, parity,
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
  cudaStream_t stream = dslashParam_->commStreams[dim * DIRECTIONS + dir];
#ifdef DEBUG
  printf("in function %s, line %d, dim = %s, dir = %s, pos = %d\n", __FUNCTION__, __LINE__, dslashDimName[dim],
         dslashDirName[dir], dim * DIRECTIONS + dir);

#endif
  // cudaStream_t stream = dslashParam_->stream1;
  void *gauge = dslashParam_->gauge;
  // void *fermionIn = dslashParam_->fermionIn;
  void *fermionIn = dslashParam_->memPool->d_recv_buffer[dim][dir];
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
      calculateBackBoundaryX<<<gridSize, blockSize_, 0, stream>>>(fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                  static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryX<<<gridSize, blockSize_, 0, stream>>>(gauge, fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                   static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == Y_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryY<<<gridSize, blockSize_, 0, stream>>>(fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                  static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryY<<<gridSize, blockSize_, 0, stream>>>(gauge, fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                   static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == Z_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryZ<<<gridSize, blockSize_, 0, stream>>>(fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                  static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryZ<<<gridSize, blockSize_, 0, stream>>>(gauge, fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                   static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == T_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryT<<<gridSize, blockSize_, 0, stream>>>(fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                  static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryT<<<gridSize, blockSize_, 0, stream>>>(gauge, fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                   static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else {
    assert(0);
  }
  // CHECK_CUDA(cudaStreamSynchronize(stream));
}

void WilsonDslash::dslashNcclIsendrecv(int dim) {
  int vectorLength;

  void *sendbuf;
  void *recvbuf;
  int dest, src;
  cudaStream_t stream2 = dslashParam_->stream2;

  switch (dim) {
    case X_DIM:
      vectorLength = dslashParam_->Ly * dslashParam_->Lz * dslashParam_->Lt / 2 * Ns * Nc;
      break;
    case Y_DIM:
      vectorLength = dslashParam_->Lx * dslashParam_->Lz * dslashParam_->Lt / 2 * Ns * Nc;
      break;
    case Z_DIM:
      vectorLength = dslashParam_->Lx * dslashParam_->Ly * dslashParam_->Lt / 2 * Ns * Nc;
      break;
    case T_DIM:
      vectorLength = dslashParam_->Lx * dslashParam_->Ly * dslashParam_->Lz / 2 * Ns * Nc;
      break;
    default:
      return;
  }

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
  // boundary calc and transfer on stream2
  if (dslashParam_->Nx > 1) {
    preDslash(X_DIM, FWD, daggerFlag);
    preDslash(X_DIM, BWD, daggerFlag);
  }
  if (dslashParam_->Ny > 1) {
    preDslash(Y_DIM, FWD, daggerFlag);
    preDslash(Y_DIM, BWD, daggerFlag);
  }
  if (dslashParam_->Nz > 1) {
    preDslash(Z_DIM, FWD, daggerFlag);
    preDslash(Z_DIM, BWD, daggerFlag);
  }
  if (dslashParam_->Nt > 1) {
    preDslash(T_DIM, FWD, daggerFlag);
    preDslash(T_DIM, BWD, daggerFlag);
  }
  // SYNC GPU
  // CHECK_CUDA(cudaStreamSynchronize(cudaStream1_));
  // CHECK_CUDA(cudaStreamSynchronize(cudaStream2_));
  // MPI_Isend MPI_Irecv
  CHECK_NCCL(ncclGroupStart());
  if (dslashParam_->Nx > 1) {
    dslashNcclIsendrecv(X_DIM);
  }

  if (dslashParam_->Ny > 1) {
    dslashNcclIsendrecv(Y_DIM);
  }

  if (dslashParam_->Nz > 1) {
    dslashNcclIsendrecv(Z_DIM);
  }

  if (dslashParam_->Nt > 1) {
    dslashNcclIsendrecv(T_DIM);
  }
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
  int sendLength;
  switch (dim) {
    case X_DIM:
      sendLength = dslashParam_->Ly * dslashParam_->Lz * dslashParam_->Lt / 2 * Ns * Nc;
      break;
    case Y_DIM:
      sendLength = dslashParam_->Lx * dslashParam_->Lz * dslashParam_->Lt / 2 * Ns * Nc;
      break;
    case Z_DIM:
      sendLength = dslashParam_->Lx * dslashParam_->Ly * dslashParam_->Lt / 2 * Ns * Nc;
      break;
    case T_DIM:
      sendLength = dslashParam_->Lx * dslashParam_->Ly * dslashParam_->Lz / 2 * Ns * Nc;
      break;
    default:
      break;
  }
  // for (int i = 0; i < Nd; i++) {
  {
    cudaStream_t streamFwd = dslashParam_->commStreams[dim * DIRECTIONS + FWD];
    cudaStream_t streamBwd = dslashParam_->commStreams[dim * DIRECTIONS + BWD];
    void *sendbuf;
    void *recvbuf;
    int dest, src;
    // if this dim isn't separated, continue
    // if ((dim == X_DIM && dslashParam_->Nx <= 1) || (dim == Y_DIM && dslashParam_->Ny <= 1) ||
    //     (dim == Z_DIM && dslashParam_->Nz <= 1) || (dim == T_DIM && dslashParam_->Nt <= 1)) {
    //   continue;
    // }

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

void WilsonDslash::dslashMPIWait(int dim) {
  int sendLength;
  // cudaStream_t stream1 = dslashParam_->stream1;
  switch (dim) {
    case X_DIM:
      sendLength = dslashParam_->Ly * dslashParam_->Lz * dslashParam_->Lt / 2 * Ns * Nc;
      break;
    case Y_DIM:
      sendLength = dslashParam_->Lx * dslashParam_->Lz * dslashParam_->Lt / 2 * Ns * Nc;
      break;
    case Z_DIM:
      sendLength = dslashParam_->Lx * dslashParam_->Ly * dslashParam_->Lt / 2 * Ns * Nc;
      break;
    case T_DIM:
      sendLength = dslashParam_->Lx * dslashParam_->Ly * dslashParam_->Lz / 2 * Ns * Nc;
      break;
    default:
      break;
  }

  {
    cudaStream_t streamfwd = dslashParam_->commStreams[dim * DIRECTIONS + FWD];
    cudaStream_t streambwd = dslashParam_->commStreams[dim * DIRECTIONS + BWD];
#ifdef DEBUG
    printf("in function %s, line %d, dim = %s, fwd = %d, bwd = %d\n", __FUNCTION__, __LINE__, dslashDimName[dim],
           dim * DIRECTIONS + FWD, dim * DIRECTIONS + BWD);
#endif
    CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[dim][FWD], MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[dim][BWD], MPI_STATUS_IGNORE));

    CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiSendRequest[dim][FWD], MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiSendRequest[dim][BWD], MPI_STATUS_IGNORE));

    CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[dim][FWD],
                               dslashParam_->memPool->h_recv_buffer[dim][FWD], sendLength * 2 * sizeof(double),
                               cudaMemcpyHostToDevice, streamfwd));
    CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[dim][BWD],
                               dslashParam_->memPool->h_recv_buffer[dim][BWD], sendLength * 2 * sizeof(double),
                               cudaMemcpyHostToDevice, streambwd));
  }
}

void WilsonDslash::cudaStreamBarrier() {
  // cudaStream_t stream1 = dslashParam_->stream1;
  cudaStream_t streamFwd;
  cudaStream_t streamBwd;
  for (int i = 0; i < Nd; i++) {
    if ((i == X_DIM && dslashParam_->Nx > 1) || (i == Y_DIM && dslashParam_->Ny > 1) ||
        (i == Z_DIM && dslashParam_->Nz > 1) || (i == T_DIM && dslashParam_->Nt > 1)) {
      streamFwd = dslashParam_->commStreams[i * DIRECTIONS + FWD];
      streamBwd = dslashParam_->commStreams[i * DIRECTIONS + BWD];
      CHECK_CUDA(cudaStreamSynchronize(streamFwd));
      CHECK_CUDA(cudaStreamSynchronize(streamBwd));
    }
  }
}

void WilsonDslash::preApply2() {
  int daggerFlag = dslashParam_->daggerFlag;
  if (dslashParam_->Nx > 1) {
    preDslashMPI(X_DIM, FWD, daggerFlag);
    preDslashMPI(X_DIM, BWD, daggerFlag);
  }
  if (dslashParam_->Ny > 1) {
    preDslashMPI(Y_DIM, FWD, daggerFlag);
    preDslashMPI(Y_DIM, BWD, daggerFlag);
  }
  if (dslashParam_->Nz > 1) {
    preDslashMPI(Z_DIM, FWD, daggerFlag);
    preDslashMPI(Z_DIM, BWD, daggerFlag);
  }
  if (dslashParam_->Nt > 1) {
    preDslashMPI(T_DIM, FWD, daggerFlag);
    preDslashMPI(T_DIM, BWD, daggerFlag);
  }

  // BARRIER
  cudaStreamBarrier();
  // SendRecv
  if (dslashParam_->Nx > 1) {
    dslashMPIIsendrecv(X_DIM);
  }
  if (dslashParam_->Ny > 1) {
    dslashMPIIsendrecv(Y_DIM);
  }
  if (dslashParam_->Nz > 1) {
    dslashMPIIsendrecv(Z_DIM);
  }
  if (dslashParam_->Nt > 1) {
    dslashMPIIsendrecv(T_DIM);
  }
}

void WilsonDslash::postApply2() {
  // WAIT
  CHECK_CUDA(cudaStreamSynchronize(dslashParam_->stream1));
  if (dslashParam_->Nx > 1) {
    dslashMPIWait(X_DIM);
  }
  if (dslashParam_->Ny > 1) {
    dslashMPIWait(Y_DIM);
  }
  if (dslashParam_->Nz > 1) {
    dslashMPIWait(Z_DIM);
  }
  if (dslashParam_->Nt > 1) {
    dslashMPIWait(T_DIM);
  }

  int daggerFlag = dslashParam_->daggerFlag;
  // cudaStreamBarrier();
  // calculate
  if (dslashParam_->Nx > 1) {
    postDslashMPI(X_DIM, FWD, daggerFlag);
    postDslashMPI(X_DIM, BWD, daggerFlag);

    CHECK_CUDA(cudaStreamSynchronize(dslashParam_->commStreams[X_DIM * DIRECTIONS + FWD]));
    CHECK_CUDA(cudaStreamSynchronize(dslashParam_->commStreams[X_DIM * DIRECTIONS + BWD]));
  }
  if (dslashParam_->Ny > 1) {
    postDslashMPI(Y_DIM, FWD, daggerFlag);
    postDslashMPI(Y_DIM, BWD, daggerFlag);

    CHECK_CUDA(cudaStreamSynchronize(dslashParam_->commStreams[Y_DIM * DIRECTIONS + FWD]));
    CHECK_CUDA(cudaStreamSynchronize(dslashParam_->commStreams[Y_DIM * DIRECTIONS + BWD]));
  }
  if (dslashParam_->Nz > 1) {
    postDslashMPI(Z_DIM, FWD, daggerFlag);
    postDslashMPI(Z_DIM, BWD, daggerFlag);

    CHECK_CUDA(cudaStreamSynchronize(dslashParam_->commStreams[Z_DIM * DIRECTIONS + FWD]));
    CHECK_CUDA(cudaStreamSynchronize(dslashParam_->commStreams[Z_DIM * DIRECTIONS + BWD]));
  }
  if (dslashParam_->Nt > 1) {
    postDslashMPI(T_DIM, FWD, daggerFlag);
    postDslashMPI(T_DIM, BWD, daggerFlag);

    CHECK_CUDA(cudaStreamSynchronize(dslashParam_->commStreams[T_DIM * DIRECTIONS + FWD]));
    CHECK_CUDA(cudaStreamSynchronize(dslashParam_->commStreams[T_DIM * DIRECTIONS + BWD]));
  }
  // cudaStreamBarrier();
  // CHECK_CUDA(cudaStreamSynchronize(dslashParam_->stream1));
}

END_NAMESPACE(qcu)