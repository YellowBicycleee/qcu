#pragma once
#include <qcu_macro.h>

BEGIN_NAMESPACE(qcu)

enum MemType { HOST_MEM = 0, DEVICE_MEM };

// 内存池
struct QcuMemPool {
public:
  // lattice desc
  const QcuDesc& latticeDesc;
  // memory pool
  // CUDA MEM
  void *d_send_buffer[Nd][DIRCTIONS];
  void *d_recv_buffer[Nd][DIRCTIONS];

  // HOST MEM
  void *h_send_buffer[Nd][DIRCTIONS];
  void *h_recv_buffer[Nd][DIRCTIONS];

  // TODO: memory pool allocation
  void allocate(size_t typeSize) {}
  // TODO: memory pool deallocation
  void deallocate() {}

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

  QcuMemPool(const QcuDesc& desc) : latticeDesc(desc) {}
  ~QcuMemPool() {}
};

END_NAMESPACE(qcu)