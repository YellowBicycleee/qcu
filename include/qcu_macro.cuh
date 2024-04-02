#pragma once
#define DEBUG
#define BEGIN_NAMESPACE(_) namespace _ {
#define END_NAMESPACE(_) }

enum DIMS { X_DIM = 0, Y_DIM, Z_DIM, T_DIM, Nd };

enum DIRS { BWD = 0, FWD = 1, DIRECTIONS };

enum PRECONDITION { PRECONDITION_OFF = 0, EVEN_ODD_PRECONDITION };

enum MemoryStorage {
  NON_COALESCED = 0,
  COALESCED = 1,
};

enum ShiftDirection {
  TO_COALESCE = 0,
  TO_NON_COALESCE = 1,
};

constexpr int Nc = 3;
constexpr int Ns = 4;

BEGIN_NAMESPACE(qcu)
struct QcuDesc {
  int lattice_size[4];
  int grid_size[4];
  QcuDesc(int Lx, int Ly, int Lz, int Lt, int Nx, int Ny, int Nz, int Nt) {
    lattice_size[0] = Lx;
    lattice_size[1] = Ly;
    lattice_size[2] = Lz;
    lattice_size[3] = Lt;

    grid_size[0] = Nx;
    grid_size[1] = Ny;
    grid_size[2] = Nz;
    grid_size[3] = Nt;
  }
};

END_NAMESPACE(qcu)

#define CHECK_MPI(cmd)                                                                             \
  do {                                                                                             \
    int err = cmd;                                                                                 \
    if (err != MPI_SUCCESS) {                                                                      \
      fprintf(stderr, "MPI error: %d\n", err);                                                     \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

#define CHECK_CUDA(cmd)                                                                            \
  do {                                                                                             \
    cudaError_t err = cmd;                                                                         \
    if (err != cudaSuccess) {                                                                      \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));                                \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

#define CHECK_NCCL(cmd)                                                                            \
  do {                                                                                             \
    ncclResult_t err = cmd;                                                                        \
    if (err != ncclSuccess) {                                                                      \
      fprintf(stderr, "NCCL error: %s\n", ncclGetErrorString(err));                                \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)
