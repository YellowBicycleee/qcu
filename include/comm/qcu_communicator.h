#pragma once

#include <assert.h>
#include <cuda.h>
#include <mpi.h>
#ifdef USE_NCCL
#include <nccl.h>
#endif

#include <cstdio>

#include "qcu_macro.cuh"

enum CommOption { USE_MPI = 0, USE_NCCL, USE_GPU_AWARE_MPI };

#define MPI_CHECK(call)                                                  \
  do {                                                                   \
    int e = call;                                                        \
    if (e != MPI_SUCCESS) {                                              \
      fprintf(stderr, "MPI error %d at %s:%d\n", e, __FILE__, __LINE__); \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

// void Qcu_MPI_Wait_gpu_aware(MPI_Request *request, MPI_Status *status);
// void Qcu_MPI_Irecv_gpu_aware(void *buf, int count, MPI_Datatype datatype, int source, int tag,
//                              MPI_Comm comm, MPI_Request *request);
// void Qcu_MPI_Isend_gpu_aware(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
//                              MPI_Comm comm, MPI_Request *request);
// void Qcu_MPI_Allreduce_gpu_aware(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
//                                  MPI_Op op, MPI_Comm comm);

BEGIN_NAMESPACE(qcu)

struct MsgHandler {
  CommOption opt;

  MPI_Request mpiSendRequest[Nd][DIRECTIONS];  // MPI_Request
  MPI_Request mpiRecvRequest[Nd][DIRECTIONS];  // MPI_Request

  MPI_Status mpiSendStatus[Nd][DIRECTIONS];  // MPI_Status
  MPI_Status mpiRecvStatus[Nd][DIRECTIONS];

#ifdef USE_NCCL
  // NCCL member
  ncclComm_t ncclComm;
  ncclUniqueId ncclId;
#endif

  // TODO: change to option Init? (consider if necessary)
  MsgHandler(CommOption opt = USE_NCCL) : opt(opt) { /*initNccl();*/
  }

  // msgSendInit 和msgRecvInit函数为外部留，当前设计不利于一次性初始化
  void msgSendInit(int dim, int fwdRank, int bwdRank, int complexLength, void* sendBufFWD, void* sendBufBWD) {
    CHECK_MPI(MPI_Send_init(sendBufFWD, complexLength * 2, MPI_DOUBLE, fwdRank, FWD, MPI_COMM_WORLD,
                            &mpiSendRequest[dim][FWD]));
    CHECK_MPI(MPI_Send_init(sendBufBWD, complexLength * 2, MPI_DOUBLE, bwdRank, BWD, MPI_COMM_WORLD,
                            &mpiSendRequest[dim][BWD]));
  }
  void msgRecvInit(int dim, int fwdRank, int bwdRank, int complexLength, void* recvBufFWD, void* recvBufBWD) {
    CHECK_MPI(MPI_Recv_init(recvBufFWD, complexLength * 2, MPI_DOUBLE, fwdRank, BWD, MPI_COMM_WORLD,
                            &mpiRecvRequest[dim][FWD]));
    CHECK_MPI(MPI_Recv_init(recvBufBWD, complexLength * 2, MPI_DOUBLE, bwdRank, FWD, MPI_COMM_WORLD,
                            &mpiRecvRequest[dim][BWD]));
  }

  ~MsgHandler() { /*destroyNccl();*/
  }

 private:
  // void initNccl();
  // void destroyNccl();
};

class QcuComm {
 private:
  int processRank;                    // rank of current process
  int numProcess;                     // number of total processes
  int processCoord[Nd];               // coord of process in the grid
  int comm_grid_size[Nd];             // int[4]     = {Nx, Ny, Nz, Nt}
  int neighbor_rank[Nd][DIRECTIONS];  // int[4][2]
  cudaIpcMemHandle_t p2p_handles[Nd][DIRECTIONS];
  int calcAdjProcess(int dim, int dir);

 public:
  // force to use the constructor
  QcuComm(int Nx, int Ny, int Nz, int Nt);
  int getNeighborRank(int dim, int dir) {
    if (dim < 0 || dim >= Nd || dir < 0 || dir >= DIRECTIONS) {
      printf("Invalid dim or dir\n");
      return -1;
    }
    return neighbor_rank[dim][dir];
  }
  ~QcuComm() {}
};

END_NAMESPACE(qcu)