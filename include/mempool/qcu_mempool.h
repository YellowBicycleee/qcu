#pragma once

#include "qcu_macro.cuh"

BEGIN_NAMESPACE(qcu)

enum MemType { HOST_MEM = 0, DEVICE_MEM };

// 内存池
struct QcuMemPool {
 public:
  // constexpr static int reduceBufferNum = 2;
  // void *h_reduce_buffer[reduceBufferNum];
  // lattice desc
  // const QcuDesc& latticeDesc;
  // memory pool
  // CUDA MEM
  void *d_send_buffer[Nd][DIRECTIONS];
  void *d_recv_buffer[Nd][DIRECTIONS];

  // HOST MEM
  void *h_send_buffer[Nd][DIRECTIONS];
  void *h_recv_buffer[Nd][DIRECTIONS];

  // p2p remote buffer
  void *p2p_send_buffer[Nd][DIRECTIONS];
  void *p2p_recv_buffer[Nd][DIRECTIONS];

  // get send bufer pointer
  void *getSendBuffer(MemType memType, int dim, int dir) {
    if (memType == HOST_MEM) {
      return h_send_buffer[dim][dir];
    } else if (memType == DEVICE_MEM) {
      return d_send_buffer[dim][dir];
    } else {  // error
      return nullptr;
    }
  }
  // get recv buffer pointer
  void *getRecvBuffer(MemType memType, int dim, int dir) {
    if (memType == HOST_MEM) {
      return h_recv_buffer[dim][dir];
    } else if (memType == DEVICE_MEM) {
      return d_recv_buffer[dim][dir];
    } else {  // error
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

    // for (int dim = 0; dim < Nd; dim++) {
    //   for (int dir = 0; dir < DIRECTIONS; dir++) {
    //     d_send_buffer[dim][dir] = nullptr;
    //     d_recv_buffer[dim][dir] = nullptr;
    //     h_send_buffer[dim][dir] = nullptr;
    //     h_recv_buffer[dim][dir] = nullptr;
    //   }
    // }
  }

  void allocateAllVector(int xDimLength, int yDimLength, int zDimLength, int tDimLength, size_t typeSize) {
    if (xDimLength > 0) allocateVector(X_DIM, typeSize, xDimLength);
    if (yDimLength > 0) allocateVector(Y_DIM, typeSize, yDimLength);
    if (zDimLength > 0) allocateVector(Z_DIM, typeSize, zDimLength);
    if (tDimLength > 0) allocateVector(T_DIM, typeSize, tDimLength);

    // for (int i = 0; i < reduceBufferNum; i++) {
    //   CHECK_CUDA(cudaMallocHost(&h_reduce_buffer[i], sizeof(double * 2)));
    // }
  }
  // TODO: memory pool allocation
  void allocateVector(int dim, size_t typeSize, size_t length) {
    if (dim < 0 || dim >= Nd || typeSize <= 0 || length <= 0) {
      return;
    }
    for (int dir = 0; dir < DIRECTIONS; dir++) {
      // HOST MEM
      CHECK_CUDA(cudaMallocHost(&h_send_buffer[dim][dir], typeSize * length));
      CHECK_CUDA(cudaMallocHost(&h_recv_buffer[dim][dir], typeSize * length));
      // DEVICE MEM
      CHECK_CUDA(cudaMalloc(&d_send_buffer[dim][dir], typeSize * length));
      CHECK_CUDA(cudaMalloc(&d_recv_buffer[dim][dir], typeSize * length));
      // P2P REMOTE MEM
      CHECK_CUDA(cudaMalloc(&p2p_send_buffer[dim][dir], typeSize * length));
      CHECK_CUDA(cudaMalloc(&p2p_recv_buffer[dim][dir], typeSize * length));
#ifdef DEBUG
      printf("size = %lu send / recv buffer allocated\ndim = %d, sendbuffer = %p, recvbuffer = %p\n", typeSize * length,
             dim, d_send_buffer[dim][dir], d_recv_buffer[dim][dir]);
#endif
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
        CHECK_CUDA(cudaFreeHost(h_send_buffer[dim][dir]));
        h_send_buffer[dim][dir] = nullptr;
      }
      if (h_recv_buffer[dim][dir] != nullptr) {
        CHECK_CUDA(cudaFreeHost(h_recv_buffer[dim][dir]));
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
      // P2P REMOTE MEM
      if (p2p_send_buffer[dim][dir] != nullptr) {
        CHECK_CUDA(cudaFree(p2p_send_buffer[dim][dir]));
        p2p_send_buffer[dim][dir] = nullptr;
      }
      if (p2p_recv_buffer[dim][dir] != nullptr) {
        CHECK_CUDA(cudaFree(p2p_recv_buffer[dim][dir]));
        p2p_recv_buffer[dim][dir] = nullptr;
      }
    }
  }

  void *getHostSendBuffer(int dim, int dir) {
    if (dim < 0 || dim >= Nd || dir < 0 || dir >= DIRECTIONS) {
      return nullptr;
    }
    return h_send_buffer[dim][dir];
  }
  void *getHostRecvBuffer(int dim, int dir) {
    if (dim < 0 || dim >= Nd || dir < 0 || dir >= DIRECTIONS) {
      return nullptr;
    }
    return h_recv_buffer[dim][dir];
  }
  void *getDeviceSendBuffer(int dim, int dir) {
    if (dim < 0 || dim >= Nd || dir < 0 || dir >= DIRECTIONS) {
      return nullptr;
    }
    return d_send_buffer[dim][dir];
  }
  void *getDeviceRecvBuffer(int dim, int dir) {
    if (dim < 0 || dim >= Nd || dir < 0 || dir >= DIRECTIONS) {
      return nullptr;
    }
    return d_recv_buffer[dim][dir];
  }
  void *getP2PSendBuffer(int dim, int dir) {
    if (dim < 0 || dim >= Nd || dir < 0 || dir >= DIRECTIONS) {
      return nullptr;
    }
    return p2p_send_buffer[dim][dir];
  }
  void *getP2PRecvBuffer(int dim, int dir) {
    if (dim < 0 || dim >= Nd || dir < 0 || dir >= DIRECTIONS) {
      return nullptr;
    }
    return p2p_recv_buffer[dim][dir];
  }

  ~QcuMemPool() {
    for (int i = 0; i < Nd; i++) {
      deallocateVector(i);
    }
  }
};

END_NAMESPACE(qcu)