#include "comm/qcu_communicator.h"

BEGIN_NAMSPACE(qcu)

QcuComm::QcuComm(int Nx, int Ny, int Nz, int Nt) {
  comm_grid_size[0] = Nx;
  comm_grid_size[1] = Ny;
  comm_grid_size[2] = Nz;
  comm_grid_size[3] = Nt;

  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &processRank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numProcess));

  processCoord[3] = processRank % Nt;
  processCoord[2] = processRank / Nt % Nz;
  processCoord[1] = processRank / Nt / Nz % Ny;
  processCoord[0] = processRank / Nt / Nz / Ny;
  printf("[pid=%d], (%d, %d, %d, %d)\n", processRank, processCoord[0],
         processCoord[1], processCoord[2], processCoord[3]);

  {
    for (int i = 0; i < Nd; i++) {
      for (int j = 0; j < DIRECTIONS; j++) {
        neighbor_rank[i][j] = calcAdjProcess(i, j);
        printf("[pid = %d], DIM = %d, DIR = %d, neighbor = %d\n", processRank,
               i, j, neighbor_rank[i][j]);
      }
    }
  }
}

int QcuComm::calcAdjProcess(int dim, int dir) {
  assert((dim == X_DIM || dim == Y_DIM || dim == Z_DIM || dim == T_DIM) &&
         (dir == FWD || dir == BWD));
  int temp[Nd];
  for (int i = 0; i < Nd; i++) {
    temp[i] = processCoord[i];
  }
  temp[dim] = (dir == FWD) ? ((processCoord[dim] + 1) % comm_grid_size[dim])
                           : ((processCoord[dim] + comm_grid_size[dim] - 1) %
                              comm_grid_size[dim]);
  return ((temp[0] * comm_grid_size[1] + temp[1]) * comm_grid_size[2] +
          temp[2]) *
             comm_grid_size[3] +
         temp[3];
}
END_NAMESPACE(qcu)