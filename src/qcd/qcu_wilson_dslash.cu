#include "qcd/qcu_wilson_dslash.cuh"
#include "targets/wilson_dslash_ghost_kernel.cuh"
#include "targets/wilson_dslash_kernel.cuh"
#include <mpi.h>
// #define DEBUG
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
  // printf("Apply Done\n");
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
  cudaStream_t stream2 = dslashParam_->stream2;
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
      calculateBackBoundaryX<<<gridSize, blockSize_, 0, stream2>>>(fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                   static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryX<<<gridSize, blockSize_, 0, stream2>>>(gauge, fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                    static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == Y_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryY<<<gridSize, blockSize_, 0, stream2>>>(fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                   static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryY<<<gridSize, blockSize_, 0, stream2>>>(gauge, fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                    static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == Z_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryZ<<<gridSize, blockSize_, 0, stream2>>>(fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                   static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryZ<<<gridSize, blockSize_, 0, stream2>>>(gauge, fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                    static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else if (dim == T_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryT<<<gridSize, blockSize_, 0, stream2>>>(fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                   static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryT<<<gridSize, blockSize_, 0, stream2>>>(gauge, fermionOut, Lx, Ly, Lz, Lt, parity,
                                                                    static_cast<Complex *>(fermionIn), daggerParam);
    }
  } else {
    assert(0);
  }
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
  cudaStream_t stream1 = dslashParam_->stream1;
  CHECK_CUDA(cudaStreamSynchronize(stream1));
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

  // #ifdef DEBUG
  //   printf("then, Begin postDslash\n");
  // #endif
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
  for (int i = 0; i < DIRECTIONS; i++) {
    cudaStream_t stream = dslashParam_->commStreams[dim * DIRECTIONS + i];
    void *sendbuf;
    void *recvbuf;
    int dest, src;
    switch (i) {
      case FWD:
        sendbuf = dslashParam_->memPool->h_send_buffer[dim][FWD];
        recvbuf = dslashParam_->memPool->h_recv_buffer[dim][BWD];
        dest = dslashParam_->qcuComm->getNeighborRank(dim, FWD);
        src = dslashParam_->qcuComm->getNeighborRank(dim, BWD);
        CHECK_MPI(MPI_Isend(sendbuf, sendLength * 2, MPI_DOUBLE, dest, FWD, MPI_COMM_WORLD,
                            &dslashParam_->msgHandler->mpiSendRequest[dim][i]));
        CHECK_MPI(MPI_Irecv(recvbuf, sendLength * 2, MPI_DOUBLE, src, FWD, MPI_COMM_WORLD,
                            &dslashParam_->msgHandler->mpiSendRequest[dim][i]));
        break;
      case BWD:
        sendbuf = dslashParam_->memPool->h_send_buffer[dim][BWD];
        recvbuf = dslashParam_->memPool->h_recv_buffer[dim][FWD];
        dest = dslashParam_->qcuComm->getNeighborRank(dim, BWD);
        src = dslashParam_->qcuComm->getNeighborRank(dim, FWD);
        CHECK_MPI(MPI_Isend(sendbuf, sendLength * 2, MPI_DOUBLE, dest, BWD, MPI_COMM_WORLD,
                            &dslashParam_->msgHandler->mpiSendRequest[dim][i]));
        CHECK_MPI(MPI_Irecv(recvbuf, sendLength * 2, MPI_DOUBLE, src, BWD, MPI_COMM_WORLD,
                            &dslashParam_->msgHandler->mpiSendRequest[dim][i]));
        break;
      default:
        return;
    }
  }
}

void WilsonDslash::dslashMPIWait(int dim) {
  int sendLength;
  if (dslashParam_->Nx > 1) {
    sendLength = dslashParam_->Ly * dslashParam_->Lz * dslashParam_->Lt / 2 * Ns * Nc;
    CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[X_DIM][FWD], MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[X_DIM][BWD], MPI_STATUS_IGNORE));
    cudaStream_t streamXF = dslashParam_->commStreams[X_DIM * DIRECTIONS + FWD];
    cudaStream_t streamXB = dslashParam_->commStreams[X_DIM * DIRECTIONS + BWD];
    CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[X_DIM][FWD],
                               dslashParam_->memPool->h_recv_buffer[X_DIM][FWD], sendLength * 2 * sizeof(double),
                               cudaMemcpyHostToDevice, streamXF));
    CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[X_DIM][BWD],
                               dslashParam_->memPool->h_recv_buffer[X_DIM][BWD], sendLength * 2 * sizeof(double),
                               cudaMemcpyHostToDevice, streamXB));
  }
  if (dslashParam_->Ny > 1) {
    sendLength = dslashParam_->Lx * dslashParam_->Lz * dslashParam_->Lt / 2 * Ns * Nc;
    CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[Y_DIM][FWD], MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[Y_DIM][BWD], MPI_STATUS_IGNORE));
    cudaStream_t streamYF = dslashParam_->commStreams[Y_DIM * DIRECTIONS + FWD];
    cudaStream_t streamYB = dslashParam_->commStreams[Y_DIM * DIRECTIONS + BWD];
    CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[Y_DIM][FWD],
                               dslashParam_->memPool->h_recv_buffer[Y_DIM][FWD], sendLength * 2 * sizeof(double),
                               cudaMemcpyHostToDevice, streamYF));
    CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[Y_DIM][BWD],
                               dslashParam_->memPool->h_recv_buffer[Y_DIM][BWD], sendLength * 2 * sizeof(double),
                               cudaMemcpyHostToDevice, streamYB));
  }
  if (dslashParam_->Nz > 1) {
    sendLength = dslashParam_->Lx * dslashParam_->Ly * dslashParam_->Lt / 2 * Ns * Nc;
    CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[Z_DIM][FWD], MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[Z_DIM][BWD], MPI_STATUS_IGNORE));
    cudaStream_t streamZF = dslashParam_->commStreams[Z_DIM * DIRECTIONS + FWD];
    cudaStream_t streamZB = dslashParam_->commStreams[Z_DIM * DIRECTIONS + BWD];
    CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[Z_DIM][FWD],
                               dslashParam_->memPool->h_recv_buffer[Z_DIM][FWD], sendLength * 2 * sizeof(double),
                               cudaMemcpyHostToDevice, streamZF));
    CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[Z_DIM][BWD],
                               dslashParam_->memPool->h_recv_buffer[Z_DIM][BWD], sendLength * 2 * sizeof(double),
                               cudaMemcpyHostToDevice, streamZB));
  }
  if (dslashParam_->Nt > 1) {
    sendLength = dslashParam_->Lx * dslashParam_->Ly * dslashParam_->Lz / 2 * Ns * Nc;
    CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[T_DIM][FWD], MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_Wait(&dslashParam_->msgHandler->mpiRecvRequest[T_DIM][BWD], MPI_STATUS_IGNORE));
    cudaStream_t streamTF = dslashParam_->commStreams[T_DIM * DIRECTIONS + FWD];
    cudaStream_t streamTB = dslashParam_->commStreams[T_DIM * DIRECTIONS + BWD];
    CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[T_DIM][FWD],
                               dslashParam_->memPool->h_recv_buffer[T_DIM][FWD], sendLength * 2 * sizeof(double),
                               cudaMemcpyHostToDevice, streamTF));
    CHECK_CUDA(cudaMemcpyAsync(dslashParam_->memPool->d_recv_buffer[T_DIM][BWD],
                               dslashParam_->memPool->h_recv_buffer[T_DIM][BWD], sendLength * 2 * sizeof(double),
                               cudaMemcpyHostToDevice, streamTB));
  }
}

void WilsonDslash::MemcpyBarrier() {
  if (dslashParam_->Nx > 1) {
    cudaStream_t streamXF = dslashParam_->commStreams[X_DIM * DIRECTIONS + FWD];
    cudaStream_t streamXB = dslashParam_->commStreams[X_DIM * DIRECTIONS + BWD];
    CHECK_CUDA(cudaStreamSynchronize(streamXF));
    CHECK_CUDA(cudaStreamSynchronize(streamXB));
  }
  if (dslashParam_->Ny > 1) {
    cudaStream_t streamYF = dslashParam_->commStreams[Y_DIM * DIRECTIONS + FWD];
    cudaStream_t streamYB = dslashParam_->commStreams[Y_DIM * DIRECTIONS + BWD];
    CHECK_CUDA(cudaStreamSynchronize(streamYF));
    CHECK_CUDA(cudaStreamSynchronize(streamYB));
  }
  if (dslashParam_->Nz > 1) {
    cudaStream_t streamZF = dslashParam_->commStreams[Z_DIM * DIRECTIONS + FWD];
    cudaStream_t streamZB = dslashParam_->commStreams[Z_DIM * DIRECTIONS + BWD];
    CHECK_CUDA(cudaStreamSynchronize(streamZF));
    CHECK_CUDA(cudaStreamSynchronize(streamZB));
  }
  if (dslashParam_->Nt > 1) {
    cudaStream_t streamTF = dslashParam_->commStreams[T_DIM * DIRECTIONS + FWD];
    cudaStream_t streamTB = dslashParam_->commStreams[T_DIM * DIRECTIONS + BWD];
    CHECK_CUDA(cudaStreamSynchronize(streamTF));
    CHECK_CUDA(cudaStreamSynchronize(streamTB));
  }
}

void WilsonDslash::preApply2() {
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
  // BARRIER
  MemcpyBarrier();
   
}

void WilsonDslash::postApply2() {
  // WAIT
  if (dslashParam_->Nx > 1) {
    dslashMPIWait(X_DIM);
    
  }
  int daggerFlag = dslashParam_->daggerFlag;

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

}
// void WilsonDslash::postDslashMPI(int dim, int dir, int daggerFlag = 0) {

// }

END_NAMESPACE(qcu)