#include "qcu_storage/qcu_storage.cuh"
#include "targets/shift_storage/qcu_shift_storage.cuh"

// TODO: non coalesced
void shiftCloverStorageTwoDouble(void *dst_vec, void *src_vec, int shift_direction, int Lx, int Ly,
                                 int Lz, int Lt) {
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol / 2;

  int block_size = BLOCK_SIZE;
  int grid_size = (half_vol + block_size - 1) / block_size;

  if (shift_direction == TO_COALESCE) {
    shift_clover_to_coalesced<<<grid_size, block_size>>>(dst_vec, src_vec, Lx, Ly, Lz, Lt);
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}



void shiftVectorStorageTwoDouble(void *dst_vec, void *src_vec, int shift_direction, int Lx, int Ly,
                                 int Lz, int Lt) {
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol / 2;

  int block_size = BLOCK_SIZE;
  int grid_size = (half_vol + block_size - 1) / block_size;

  if (shift_direction == TO_COALESCE) {
    shift_vector_to_coalesed<<<grid_size, block_size>>>(dst_vec, src_vec, Lx, Ly, Lz, Lt);
    CHECK_CUDA(cudaDeviceSynchronize());
  } else {
    shift_vector_to_noncoalesed<<<grid_size, block_size>>>(dst_vec, src_vec, Lx, Ly, Lz, Lt);
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}


void shiftGaugeStorageTwoDouble(void *dst_vec, void *src_vec, int shift_direction, int Lx, int Ly,
                                int Lz, int Lt) {
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol / 2;

  int block_size = BLOCK_SIZE;
  int grid_size = (half_vol + block_size - 1) / block_size;

  if (shift_direction == TO_COALESCE) {
    shift_gauge_to_coalesed<<<grid_size, block_size>>>(dst_vec, src_vec, Lx, Ly, Lz, Lt);
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}