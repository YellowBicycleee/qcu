#pragma once
#include "qcu_macro.cuh"

BEGIN_NAMESPACE(qcu)

enum MemType { HOST_MEM = 0, DEVICE_MEM };

// 内存池
struct QcuMemPool {
public:
  // lattice desc
  // const QcuDesc& latticeDesc;
  // memory pool
  // CUDA MEM
  void *d_send_buffer[Nd][DIRECTIONS];
  void *d_recv_buffer[Nd][DIRECTIONS];

  // HOST MEM
  void *h_send_buffer[Nd][DIRECTIONS];
  void *h_recv_buffer[Nd][DIRECTIONS];

  // Host Gauge Buffer
  // void *h_gauge_buffer[DIRECTIONS][2]; // 2: send=0, recv=1
  // Device Gauge Buffer
  // void *d_gauge_buffer[DIRECTIONS][2]; // 2: send=0, recv=1

  // get send bufer pointer
  void *getSendBuffer(MemType memType, int dim, int dir) {
    if (memType == HOST_MEM) {
      return h_send_buffer[dim][dir];
    } else if (memType == DEVICE_MEM) {
      return d_send_buffer[dim][dir];
    } else { // error
      return nullptr;
    }
  }
  // get recv buffer pointer
  void *getRecvBuffer(MemType memType, int dim, int dir) {
    if (memType == HOST_MEM) {
      return h_recv_buffer[dim][dir];
    } else if (memType == DEVICE_MEM) {
      return d_recv_buffer[dim][dir];
    } else { // error
      return nullptr;
    }
  }

  // QcuMemPool(const QcuDesc& desc) : latticeDesc(desc) {}
  QcuMemPool() {
    for (int dim = 0; dim < Nd; dim++) {
      for (int dir = 0; dir < DIRECTIONS; dir++) {
        d_send_buffer[dim][dir] = nullptr;
        d_recv_buffer[dim][dir] = nullptr;
        h_send_buffer[dim][dir] = nullptr;
        h_recv_buffer[dim][dir] = nullptr;
      }
    }

    for (int dim = 0; dim < Nd; dim++) {
      for (int dir = 0; dir < DIRECTIONS; dir++) {
        d_send_buffer[dim][dir] = nullptr;
        d_recv_buffer[dim][dir] = nullptr;
        h_send_buffer[dim][dir] = nullptr;
        h_recv_buffer[dim][dir] = nullptr;
      }
    }
  }

  void allocateAllVector(int xDimLength, int yDimLength, int zDimLength, int tDimLength, size_t typeSize) {
    if (xDimLength > 0) allocateVector(X_DIM, typeSize, xDimLength);
    if (yDimLength > 0) allocateVector(Y_DIM, typeSize, yDimLength);
    if (zDimLength > 0) allocateVector(Z_DIM, typeSize, zDimLength);
    if (tDimLength > 0) allocateVector(T_DIM, typeSize, tDimLength);
  }
  // TODO: memory pool allocation
  void allocateVector(int dim, size_t typeSize, size_t length) {
    if (dim < 0 || dim >= Nd || typeSize <= 0 || length <= 0) {
      return;
    }
    for (int dir = 0; dir < DIRECTIONS; dir++) {
      // HOST MEM
      CHECK_CUDA(cudaMalloc(&d_send_buffer[dim][dir], typeSize * length));
      CHECK_CUDA(cudaMalloc(&d_recv_buffer[dim][dir], typeSize * length));
      // DEVICE MEM
      cudaMalloc(&d_send_buffer[dim][dir], typeSize * length);
      cudaMalloc(&d_recv_buffer[dim][dir], typeSize * length);
    }
  }
  // TODO: memory pool deallocation
  void deallocateVector(int dim) {
    if (dim < 0 || dim >= Nd) {
      return;
    }
    for (int dir = 0; dir < DIRECTIONS; dir++) {
      // HOST MEM
      if (h_send_buffer[dim][dir] != nullptr) {
        CHECK_CUDA(cudaFree(h_send_buffer[dim][dir]));
        h_send_buffer[dim][dir] = nullptr;
      }
      if (h_recv_buffer[dim][dir] != nullptr) {
        CHECK_CUDA(cudaFree(h_recv_buffer[dim][dir]));
        h_recv_buffer[dim][dir] = nullptr;
      }
      // DEVICE MEM
      if (d_send_buffer[dim][dir] != nullptr) {
        CHECK_CUDA(cudaFree(d_send_buffer[dim][dir]));
        d_send_buffer[dim][dir] = nullptr;
      }
      if (d_recv_buffer[dim][dir] != nullptr) {
        CHECK_CUDA(cudaFree(d_recv_buffer[dim][dir]));
        d_recv_buffer[dim][dir] = nullptr;
      }
    }
  }

  void* getHostSendBuffer(int dim, int dir) {
    if (dim < 0 || dim >= Nd || dir < 0 || dir >= DIRECTIONS) {
      return nullptr;
    }
    return h_send_buffer[dim][dir];
  }
  void* getHostRecvBuffer(int dim, int dir) {
    if (dim < 0 || dim >= Nd || dir < 0 || dir >= DIRECTIONS) {
      return nullptr;
    }
    return h_recv_buffer[dim][dir];
  }
  void* getDeviceSendBuffer(int dim, int dir) {
    if (dim < 0 || dim >= Nd || dir < 0 || dir >= DIRECTIONS) {
      return nullptr;
    }
    return d_send_buffer[dim][dir];
  }
  void* getDeviceRecvBuffer(int dim, int dir) {
    if (dim < 0 || dim >= Nd || dir < 0 || dir >= DIRECTIONS) {
      return nullptr;
    }
    return d_recv_buffer[dim][dir];
  }

  // void* getHostGaugeSendBuffer(int dim) {
  //   if (dim < 0 || dim >= Nd) {
  //     return nullptr;
  //   }
  //   return h_gauge_buffer[dim][0];
  // }
  // void* getHostGaugeRecvBuffer(int dim) {
  //   if (dim < 0 || dim >= Nd) {
  //     return nullptr;
  //   }
  //   return h_gauge_buffer[dim][1];
  // }
  // void* getDeviceGaugeSendBuffer(int dim) {
  //   if (dim < 0 || dim >= Nd) {
  //     return nullptr;
  //   }
  //   return d_gauge_buffer[dim][0];
  // }
  // void* getDeviceGaugeRecvBuffer(int dim) {
  //   if (dim < 0 || dim >= Nd) {
  //     return nullptr;
  //   }
  //   return d_gauge_buffer[dim][1];
  // }

  ~QcuMemPool() {
    for (int i = 0; i < Nd; i++) {
      deallocateVector(i);
    }
  }
};

END_NAMESPACE(qcu)