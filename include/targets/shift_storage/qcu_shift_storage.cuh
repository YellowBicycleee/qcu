#pragma once
// #include "qcu_shift_storage_complex.cuh"
// #include <cstdio>
#include "qcu_macro.cuh"

// enum MemoryStorage {
//   NON_COALESCED = 0,
//   COALESCED = 1,
// };

// enum ShiftDirection {
//   TO_COALESCE = 0,
//   TO_NON_COALESCE = 1,
// };


// ----
// #include "qcu_shift_storage_complex.cuh"
#include <cstdio>
#include "qcu_macro.cuh"


#define WARP_SIZE 32
#define BLOCK_SIZE 256

// DONE: WARP version, no sync  
static __device__ __forceinline__ void storeVectorBySharedMemory(void* origin, void* result) {
  __shared__ double shared_buffer[BLOCK_SIZE * Ns * Nc * 2];
  int thread = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_index = (thread - thread / BLOCK_SIZE * BLOCK_SIZE) / WARP_SIZE;//thread % BLOCK_SIZE / WARP_SIZE;

  // result is register variable
  double* shared_dst = shared_buffer + threadIdx.x * Ns * Nc * 2;
  double* warp_src = static_cast<double*>(origin) + (thread / WARP_SIZE * WARP_SIZE) * Ns * Nc * 2;


  // load data to register
  double* register_addr = static_cast<double*>(result);
  for (int i = 0; i < Ns * Nc * 2; i++) {
    shared_dst[i] = register_addr[i];
  }

  // store result of shared memory to global memory
  for (int i = threadIdx.x - threadIdx.x / WARP_SIZE * WARP_SIZE; i < WARP_SIZE * Ns * Nc * 2; i += WARP_SIZE) {
    warp_src[i] = shared_buffer[warp_index * WARP_SIZE * Ns * Nc * 2 + i];
  }
}

// DONE: WARP version, no sync  
static __device__ __forceinline__ void loadVectorBySharedMemory(void* origin, void* result) {
  __shared__ double shared_buffer[BLOCK_SIZE * Ns * Nc * 2];
  int thread = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_index = (thread - thread / BLOCK_SIZE * BLOCK_SIZE) / WARP_SIZE; //thread % BLOCK_SIZE / WARP_SIZE;

  // result is register variable
  double* shared_dst = shared_buffer + threadIdx.x * Ns * Nc * 2;
  double* warp_src = static_cast<double*>(origin) + (thread / WARP_SIZE * WARP_SIZE) * Ns * Nc * 2;

  // store result of shared memory to global memory
  for (int i = threadIdx.x - threadIdx.x / WARP_SIZE * WARP_SIZE; i < WARP_SIZE * Ns * Nc * 2; i += WARP_SIZE) {
    shared_buffer[warp_index * WARP_SIZE * Ns * Nc * 2 + i] = warp_src[i];
  }

  // load data to register
  double* register_addr = static_cast<double*>(result);
  for (int i = 0; i < Ns * Nc * 2; i++) {
    register_addr[i] = shared_dst[i];
  }
}


static __device__ __forceinline__ void loadGaugeBySharedMemory(void* origin, void* result) {
  __shared__ double shared_buffer[BLOCK_SIZE * Nc * Nc * 2];
  int thread = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_index = (thread - thread / BLOCK_SIZE * BLOCK_SIZE) / WARP_SIZE;//thread % BLOCK_SIZE / WARP_SIZE;

  // result is register variable
  double* shared_dst = shared_buffer + threadIdx.x * Nc * Nc * 2;
  double* warp_src = static_cast<double*>(origin) + (thread / WARP_SIZE * WARP_SIZE) * Nc * Nc * 2;

  // store result of shared memory to global memory
  for (int i = threadIdx.x - threadIdx.x / WARP_SIZE * WARP_SIZE; i < WARP_SIZE * Nc * Nc * 2; i += WARP_SIZE) {
    shared_buffer[warp_index * WARP_SIZE * Nc * Nc * 2 + i] = warp_src[i];
  }

  // load data to register
  double* register_addr = static_cast<double*>(result);
  for (int i = 0; i < (Nc-1) * Nc * 2; i++) {
    register_addr[i] = shared_dst[i];
  }
}


// DONE: Lx is full Lx, not Lx / 2
static __global__ void shift_vector_to_coalesed (void* dst_vec, void* src_vec, int Lx, int Ly, int Lz, int Lt) {
  // change storage to [parity, Ns, Nc, t, z, y, x, 2]
  int sub_Lx = Lx >> 1;
  int sub_vol = sub_Lx * Ly * Lz * Lt;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  // double* src_vec_pointer = static_cast<double*>(src_vec) + thread_id * Ns * Nc * 2;
  double* dst_vec_pointer = static_cast<double*>(dst_vec) + thread_id * 2;  // complex: 2 double

  // mofify 
  double data_local[Ns * Nc * 2];
  loadVectorBySharedMemory(src_vec, data_local);

  for (int i = 0; i < Ns * Nc; i++) {
    dst_vec_pointer[0] = data_local[2 * i];
    dst_vec_pointer[1] = data_local[2 * i + 1];
    dst_vec_pointer += sub_vol * 2;
  }
}


// TODO: TO optimize      Lx is full Lx, not Lx / 2
static __global__ void shift_vector_to_noncoalesed (void* dst_vec, void* src_vec, int Lx, int Ly, int Lz, int Lt) {
  // change storage to [parity, Ns, Nc, 2, t, z, y, x]
  int sub_Lx = Lx >> 1;
  int sub_vol = sub_Lx * Ly * Lz * Lt;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  // double* dst_vec_pointer = static_cast<double*>(dst_vec) + thread_id * Ns * Nc * 2;
  double* src_vec_pointer = static_cast<double*>(src_vec) + thread_id * 2;

  double local_vector[Ns * Nc * 2];

  for (int i = 0; i < Ns * Nc; i++) {
    local_vector[2 * i] = src_vec_pointer[0];
    local_vector[2 * i + 1] = src_vec_pointer[1];
    src_vec_pointer += sub_vol * 2;
  }
  storeVectorBySharedMemory(dst_vec, local_vector);

}





// Lx is full Lx, not Lx / 2
static __global__ void shift_gauge_to_coalesed (void* dst_gauge, void* src_gauge, int Lx, int Ly, int Lz, int Lt) {
  // each thread shift both even and odd part
  // change storage to [Nd, parity, Nc-1, Nc, 2, t, z, y, x/2]
  int sub_Lx = Lx >> 1;
  int sub_vol = sub_Lx * Ly * Lz * Lt;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  int t = thread_id / (Lz * Ly * sub_Lx);
  int z = thread_id % (Lz * Ly * sub_Lx) / (Ly * sub_Lx);
  int y = thread_id % (Ly * sub_Lx) / sub_Lx;
  int sub_x = thread_id % sub_Lx;

  double local_gauge[(Nc - 1) * Nc * 2];

  double* dst_gauge_ptr;
  double* src_gauge_ptr;
  for (int i = 0; i < Nd; i++) {
    for (int parity = 0; parity < 2; parity++) {
      dst_gauge_ptr = static_cast<double*>(dst_gauge) + (2 * i + parity) * sub_vol * (Nc - 1) * Nc * 2 + (((t * Lz + z) * Ly + y) * sub_Lx + sub_x) * 2;
      src_gauge_ptr = static_cast<double*>(src_gauge) + (2 * i + parity) * sub_vol * Nc * Nc * 2; //  + (((t * Lz + z) * Ly + y) * sub_Lx + sub_x) * 2 * Nc * Nc;
      loadGaugeBySharedMemory(src_gauge_ptr, local_gauge);

      // for (int j = 0 ; j < Nc * (Nc-1) * 2; j++) {
      //   *dst_gauge_ptr = local_gauge[j];
      //   dst_gauge_ptr += sub_vol;
      // }
      for (int j = 0 ; j < Nc * (Nc-1); j++) {
        dst_gauge_ptr[0] = local_gauge[2 * j];
        dst_gauge_ptr[1] = local_gauge[2 * j + 1];
        dst_gauge_ptr += sub_vol * 2;
      }
    }
  }
}



// TODO: optimize
static __global__ void shift_clover_to_coalesced (void* dst_vec, void* src_vec, \
                                                  int Lx, int Ly, int Lz, int Lt \
){

  int half_vol = Lx * Ly * Lz * Lt / 2;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  const int clover_length = Ns * Nc * Ns * Nc / 2;

  double* src_ptr;
  double* dst_ptr;

  double local_clover[clover_length * 2];
  src_ptr = static_cast<double*>(src_vec) + clover_length * thread_id * 2;
  // load
  for (int i = 0; i < clover_length; i++) {
    local_clover[2 * i] = src_ptr[2 * i];
    local_clover[2 * i + 1] = src_ptr[2 * i + 1];
  }
  // store
  dst_ptr = static_cast<double*>(dst_vec) + thread_id * 2;
  for (int i = 0; i < clover_length; i++) {
    dst_ptr[0] = local_clover[2 * i];
    dst_ptr[1] = local_clover[2 * i + 1];
    dst_ptr += 2 * half_vol;
  }
}