#pragma once

#include "qcu_macro.cuh"
#include <assert.h>
#include <cstdio>
#include <cuda.h>
#include <mpi.h>
#include <nccl.h>

enum CommOption { USE_MPI = 0, USE_NCCL, USE_GPU_AWARE_MPI };

#define MPI_CHECK(call)                                                                            \
  do {                                                                                             \
    int e = call;                                                                                  \
    if (e != MPI_SUCCESS) {                                                                        \
      fprintf(stderr, "MPI error %d at %s:%d\n", e, __FILE__, __LINE__);                           \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

void Qcu_MPI_Wait_gpu_aware(MPI_Request *request, MPI_Status *status);
void Qcu_MPI_Irecv_gpu_aware(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                             MPI_Comm comm, MPI_Request *request);
void Qcu_MPI_Isend_gpu_aware(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                             MPI_Comm comm, MPI_Request *request);
void Qcu_MPI_Allreduce_gpu_aware(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                                 MPI_Op op, MPI_Comm comm);

BEGIN_NAMESPACE(qcu)

struct MsgHandler {
  CommOption opt;

  MPI_Request mpiSendRequest[Nd][DIRECTIONS]; // MPI_Request
  MPI_Request mpiRecvRequest[Nd][DIRECTIONS]; // MPI_Request

  MPI_Status mpiSendStatus[Nd][DIRECTIONS]; // MPI_Status
  MPI_Status mpiRecvStatus[Nd][DIRECTIONS];

  // NCCL member
  ncclComm_t ncclComm;
  ncclUniqueId ncclId;

  // TODO: change to option Init? (consider if necessary)
  MsgHandler(CommOption opt = USE_GPU_AWARE_MPI) : opt(opt) { initNccl(); }
  ~MsgHandler() { destroyNccl(); }

private:
  void initNccl();
  void destroyNccl();
};

class QcuComm {
private:
  int processRank;                   // rank of current process
  int numProcess;                    // number of total processes
  int processCoord[Nd];              // coord of process in the grid
  int comm_grid_size[Nd];            // int[4]     = {Nx, Ny, Nz, Nt}
  int neighbor_rank[Nd][DIRECTIONS]; // int[4][2]

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