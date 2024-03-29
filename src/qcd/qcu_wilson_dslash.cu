#include "qcd/qcu_wilson_dslash.cuh"
#include "targets/wilson_dslash_ghost_kernel.cuh"
#include "targets/wilson_dslash_kernel.cuh"

// #define DEBUG
BEGIN_NAMESPACE(qcu)

#ifdef DEBUG
MPI_Status status[Nd][DIRECTIONS];
MPI_Request request[Nd][DIRECTIONS];
#endif

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

  double daggerParam = daggerFlag ? -1.0 : 1.0;
  int vol = Lx * Ly * Lz * Lt;

  int gridSize = (vol / 2 + blockSize_ - 1) / blockSize_;

  dslashKernelFunc<<<gridSize, blockSize_, 0, cudaStream1_>>>(
      gauge, fermionIn, fermionOut, Lx, Ly, Lz, Lt, parity, Nx, Ny, Nz, Nt, daggerParam);
  printf("Apply Done\n");
}

// only call kernel funcs
void WilsonDslash::preDslash(int dim, int dir, int daggerFlag) {
#ifdef DEBUG
  printf("PreDslash BEGIN: dim = %d, dir = %d, daggerFlag = %d\n", dim, dir, daggerFlag);
#endif
  if ((dim < X_DIM || dim > T_DIM) || (dir < 0 || dir > 1)) {
    return;
  }
  // dir == 0 ---> stream1
  // dir == 1 ---> stream2
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
      DslashTransferBackX<<<gridSize, blockSize_, 0, cudaStream2_>>>(
          fermionIn, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontX<<<gridSize, blockSize_, 0, cudaStream2_>>>(
          gauge, fermionIn, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionOut),
          daggerParam);
    }
  } else if (dim == Y_DIM) {
    if (dir == BWD) {
      DslashTransferBackY<<<gridSize, blockSize_, 0, cudaStream2_>>>(
          fermionIn, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontY<<<gridSize, blockSize_, 0, cudaStream2_>>>(
          gauge, fermionIn, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionOut),
          daggerParam);
    }

  } else if (dim == Z_DIM) {
    if (dir == BWD) {
      DslashTransferBackZ<<<gridSize, blockSize_, 0, cudaStream2_>>>(
          fermionIn, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontZ<<<gridSize, blockSize_, 0, cudaStream2_>>>(
          gauge, fermionIn, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionOut),
          daggerParam);
    }

  } else if (dim == T_DIM) {
    if (dir == BWD) {
      DslashTransferBackT<<<gridSize, blockSize_, 0, cudaStream2_>>>(
          fermionIn, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionOut));
    } else {
      DslashTransferFrontT<<<gridSize, blockSize_, 0, cudaStream2_>>>(
          gauge, fermionIn, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionOut),
          daggerParam);
    }
  } else {
    return;
  }
}

void WilsonDslash::postDslash(int dim, int dir, int daggerFlag) {

  if ((dim < X_DIM || dim > T_DIM) || (dir < 0 || dir > 1)) {
    return;
  }
  // dir == 0 ---> stream1
  // dir == 1 ---> stream2
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
      calculateBackBoundaryX<<<gridSize, blockSize_, 0, cudaStream1_>>>(
          fermionOut, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryX<<<gridSize, blockSize_, 0, cudaStream1_>>>(
          gauge, fermionOut, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionIn),
          daggerParam);
    }
  } else if (dim == Y_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryY<<<gridSize, blockSize_, 0, cudaStream1_>>>(
          fermionOut, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryY<<<gridSize, blockSize_, 0, cudaStream1_>>>(
          gauge, fermionOut, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionIn),
          daggerParam);
    }
  } else if (dim == Z_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryZ<<<gridSize, blockSize_, 0, cudaStream1_>>>(
          fermionOut, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryZ<<<gridSize, blockSize_, 0, cudaStream1_>>>(
          gauge, fermionOut, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionIn),
          daggerParam);
    }
  } else if (dim == T_DIM) {
    if (dir == BWD) {
      calculateBackBoundaryT<<<gridSize, blockSize_, 0, cudaStream1_>>>(
          fermionOut, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionIn));
    } else {
      calculateFrontBoundaryT<<<gridSize, blockSize_, 0, cudaStream1_>>>(
          gauge, fermionOut, Lx, Ly, Lz, Lt, parity, static_cast<Complex *>(fermionIn),
          daggerParam);
    }
  } else {
    return;
  }
}

void WilsonDslash::dslashMpiIsendrecv(int dim) {
  int vectorLength;

  void *sendbuf; //  = dslashParam_->memPool->d_send_buffer[dim][dir];
  void *recvbuf; // = dslashParam_->memPool->d_recv_buffer[dim][dir];
  int dest, src; //  = dslashParam_->qcuComm->getNeighborRank(dim, dir);

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
  CHECK_NCCL(ncclSend(sendbuf, vectorLength * 2, ncclDouble, dest, comm, cudaStream2_));
  CHECK_NCCL(ncclRecv(recvbuf, vectorLength * 2, ncclDouble, src, comm, cudaStream2_));
  // CHECK_NCCL(ncclGroupEnd());

  // CHECK_CUDA(cudaStreamSynchronize(cudaStream2_));
  // send to bwd, and send from fwd
  sendbuf = dslashParam_->memPool->d_send_buffer[dim][BWD];
  recvbuf = dslashParam_->memPool->d_recv_buffer[dim][FWD];
  dest = dslashParam_->qcuComm->getNeighborRank(dim, BWD);
  src = dslashParam_->qcuComm->getNeighborRank(dim, FWD);
  // CHECK_NCCL(ncclGroupStart());
  CHECK_NCCL(ncclSend(sendbuf, vectorLength * 2, ncclDouble, dest, comm, cudaStream2_));
  CHECK_NCCL(ncclRecv(recvbuf, vectorLength * 2, ncclDouble, src, comm, cudaStream2_));
  // CHECK_NCCL(ncclGroupEnd());
}

void WilsonDslash::dslashMpiWait() { CHECK_CUDA(cudaStreamSynchronize(cudaStream2_)); }

// void WilsonDslash::preApply(int daggerFlag) {
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
    dslashMpiIsendrecv(X_DIM);
  }

  if (dslashParam_->Ny > 1) {
    dslashMpiIsendrecv(Y_DIM);
  }

  if (dslashParam_->Nz > 1) {
    dslashMpiIsendrecv(Z_DIM);
  }

  if (dslashParam_->Nt > 1) {
    dslashMpiIsendrecv(T_DIM);
  }
  CHECK_NCCL(ncclGroupEnd());
#ifdef DEBUG
  printf("PreApply Done\n");
#endif
}
// void WilsonDslash::postApply(int daggerFlag) {
void WilsonDslash::postApply() {
  int daggerFlag = dslashParam_->daggerFlag;
  // MPI_Wait
  if (dslashParam_->Nx > 1 || dslashParam_->Ny > 1 || dslashParam_->Nz > 1 ||
      dslashParam_->Nt > 1) {
    dslashMpiWait();
  }

#ifdef DEBUG
  printf("then, Begin postDslash\n");
#endif
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
  printf("PostApply Done\n");
}
END_NAMESPACE(qcu)