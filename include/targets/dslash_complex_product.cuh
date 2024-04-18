#pragma once

#include "basic_data/qcu_complex.cuh"
#include "qcu_macro.cuh"
#include "targets/public_kernels.cuh"

// separate loops
template <int _dir, int _fb>  // _dir from 0-3 means X-Z
static __device__ __forceinline__ void spinor_gauge_mul_add_vec(Complex *u_local, Complex *src_local,
                                                                Complex *dst_local, double flag) {
  printf(
      "function undefined, check your template parameters, \n"
      "now _dir = %d, _fb = %d\n",
      _dir, _fb);
  // exit(-1);
  // asm("exit;");
  assert(0);
}

template <>
__device__ __forceinline__ void spinor_gauge_mul_add_vec<X_DIM, FWD>(Complex *u_local, Complex *src_local,
                                                                     Complex *dst_local, double flag) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j].multiply_i() * flag) * u_local[i * Nc + j];
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j].multiply_i() * flag) * u_local[i * Nc + j];
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[3 * Nc + i] += temp1.multiply_i() * flag;
    dst_local[1 * Nc + i] += temp2;
    dst_local[2 * Nc + i] += temp2.multiply_i() * flag;
  }
}

template <>
__device__ __forceinline__ void spinor_gauge_mul_add_vec<X_DIM, BWD>(Complex *u_local, Complex *src_local,
                                                                     Complex *dst_local, double flag) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();
#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j].multiply_i() * flag) *
               u_local[j * Nc + i].conj();  // transpose and conj

      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j].multiply_i() * flag) *
               u_local[j * Nc + i].conj();  // transpose and conj
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[3 * Nc + i] += temp1.multiply_minus_i() * flag;
    dst_local[1 * Nc + i] += temp2;
    dst_local[2 * Nc + i] += temp2.multiply_minus_i() * flag;
  }
}

template <>
__device__ __forceinline__ void spinor_gauge_mul_add_vec<Y_DIM, FWD>(Complex *u_local, Complex *src_local,
                                                                     Complex *dst_local, double flag) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j] * flag) * u_local[i * Nc + j];
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j] * flag) * u_local[i * Nc + j];
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[3 * Nc + i] += temp1 * flag;
    dst_local[1 * Nc + i] += temp2;
    dst_local[2 * Nc + i] += -temp2 * flag;
  }
}

template <>
__device__ __forceinline__ void spinor_gauge_mul_add_vec<Y_DIM, BWD>(Complex *u_local, Complex *src_local,
                                                                     Complex *dst_local, double flag) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();
#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j] * flag) *
               u_local[j * Nc + i].conj();  // transpose and conj
                                            // second row vector with col vector
      temp2 +=
          (src_local[1 * Nc + j] + src_local[2 * Nc + j] * flag) * u_local[j * Nc + i].conj();  // transpose and conj
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[3 * Nc + i] += -temp1 * flag;
    dst_local[1 * Nc + i] += temp2;
    dst_local[2 * Nc + i] += temp2 * flag;
  }
}

template <>
__device__ __forceinline__ void spinor_gauge_mul_add_vec<Z_DIM, FWD>(Complex *u_local, Complex *src_local,
                                                                     Complex *dst_local, double flag) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j].multiply_i() * flag) * u_local[i * Nc + j];
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j].multiply_i() * flag) * u_local[i * Nc + j];
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[2 * Nc + i] += temp1.multiply_i() * flag;
    dst_local[1 * Nc + i] += temp2;
    dst_local[3 * Nc + i] += temp2.multiply_minus_i() * flag;
  }
}

template <>
__device__ __forceinline__ void spinor_gauge_mul_add_vec<Z_DIM, BWD>(Complex *u_local, Complex *src_local,
                                                                     Complex *dst_local, double flag) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j].multiply_i() * flag) *
               u_local[j * Nc + i].conj();  // transpose and conj
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j].multiply_i() * flag) *
               u_local[j * Nc + i].conj();  // transpose and conj
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[2 * Nc + i] += temp1.multiply_minus_i() * flag;
    dst_local[1 * Nc + i] += temp2;
    dst_local[3 * Nc + i] += temp2.multiply_i() * flag;
  }
}

template <>
__device__ __forceinline__ void spinor_gauge_mul_add_vec<T_DIM, FWD>(Complex *u_local, Complex *src_local,
                                                                     Complex *dst_local, double flag) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j] * flag) * u_local[i * Nc + j];
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j] * flag) * u_local[i * Nc + j];
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[2 * Nc + i] += -temp1 * flag;
    dst_local[1 * Nc + i] += temp2;
    dst_local[3 * Nc + i] += -temp2 * flag;
  }
}

template <>
__device__ __forceinline__ void spinor_gauge_mul_add_vec<T_DIM, BWD>(Complex *u_local, Complex *src_local,
                                                                     Complex *dst_local, double flag) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 +=
          (src_local[0 * Nc + j] + src_local[2 * Nc + j] * flag) * u_local[j * Nc + i].conj();  // transpose and conj
      // second row vector with col vector
      temp2 +=
          (src_local[1 * Nc + j] + src_local[3 * Nc + j] * flag) * u_local[j * Nc + i].conj();  // transpose and conj
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[2 * Nc + i] += temp1 * flag;
    dst_local[1 * Nc + i] += temp2;
    dst_local[3 * Nc + i] += temp2 * flag;
  }
}

// seperate line
// ----------------------------------
// new function
template <int _dir, int _fb, int daggerFlag>  // _dir from 0-3 means X-Z
static __device__ __forceinline__ void dslashMVKernel(Complex *u_local, Complex *src_local, Complex *dst_local) {
  printf(
      "function undefined, check your template parameters, \n"
      "now _dir = %d, _fb = %d\n",
      _dir, _fb);
  // exit(-1);
  // asm("exit;");
  assert(0);
}

template <>  // _dir from 0-3 means X-Z
__device__ __forceinline__ void dslashMVKernel<X_DIM, FWD, QCU_DAGGER_NO>(Complex *u_local, Complex *src_local,
                                                                          Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j].multiply_i()) * u_local[i * Nc + j];
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j].multiply_i()) * u_local[i * Nc + j];
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[3 * Nc + i] += temp1.multiply_i();
    dst_local[1 * Nc + i] += temp2;
    dst_local[2 * Nc + i] += temp2.multiply_i();
  }
}

template <>  // _dir from 0-3 means X-Z
__device__ __forceinline__ void dslashMVKernel<X_DIM, FWD, QCU_DAGGER_YES>(Complex *u_local, Complex *src_local,
                                                                           Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j].multiply_i()) * u_local[i * Nc + j];
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j].multiply_i()) * u_local[i * Nc + j];
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[3 * Nc + i] -= temp1.multiply_i();
    dst_local[1 * Nc + i] += temp2;
    dst_local[2 * Nc + i] -= temp2.multiply_i();
  }
}

template <>  // _dir from 0-3 means X-Z
__device__ __forceinline__ void dslashMVKernel<X_DIM, BWD, QCU_DAGGER_NO>(Complex *u_local, Complex *src_local,
                                                                          Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();
#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j].multiply_i()) *
               u_local[j * Nc + i].conj();  // transpose and conj

      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j].multiply_i()) *
               u_local[j * Nc + i].conj();  // transpose and conj
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[3 * Nc + i] += temp1.multiply_minus_i();
    dst_local[1 * Nc + i] += temp2;
    dst_local[2 * Nc + i] += temp2.multiply_minus_i();
  }
}
template <>  // _dir from 0-3 means X-Z
__device__ __forceinline__ void dslashMVKernel<X_DIM, BWD, QCU_DAGGER_YES>(Complex *u_local, Complex *src_local,
                                                                           Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();
#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j].multiply_i()) *
               u_local[j * Nc + i].conj();  // transpose and conj

      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j].multiply_i()) *
               u_local[j * Nc + i].conj();  // transpose and conj
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[3 * Nc + i] -= temp1.multiply_minus_i();
    dst_local[1 * Nc + i] += temp2;
    dst_local[2 * Nc + i] -= temp2.multiply_minus_i();
  }
}

template <>  // _dir from 0-3 means X-Z
__device__ __forceinline__ void dslashMVKernel<Y_DIM, FWD, QCU_DAGGER_NO>(Complex *u_local, Complex *src_local,
                                                                          Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j]) * u_local[i * Nc + j];
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j]) * u_local[i * Nc + j];
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[3 * Nc + i] += temp1;
    dst_local[1 * Nc + i] += temp2;
    dst_local[2 * Nc + i] -= temp2;
  }
}

template <>  // _dir from 0-3 means X-Z
__device__ __forceinline__ void dslashMVKernel<Y_DIM, FWD, QCU_DAGGER_YES>(Complex *u_local, Complex *src_local,
                                                                           Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j]) * u_local[i * Nc + j];
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j]) * u_local[i * Nc + j];
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[3 * Nc + i] -= temp1;
    dst_local[1 * Nc + i] += temp2;
    dst_local[2 * Nc + i] += temp2;
  }
}
template <>
__device__ __forceinline__ void dslashMVKernel<Y_DIM, BWD, QCU_DAGGER_NO>(Complex *u_local, Complex *src_local,
                                                                          Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();
#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j]) *
               u_local[j * Nc + i].conj();  // transpose and conj
                                            // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j]) * u_local[j * Nc + i].conj();  // transpose and conj
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[3 * Nc + i] -= temp1;
    dst_local[1 * Nc + i] += temp2;
    dst_local[2 * Nc + i] += temp2;
  }
}
template <>
__device__ __forceinline__ void dslashMVKernel<Y_DIM, BWD, QCU_DAGGER_YES>(Complex *u_local, Complex *src_local,
                                                                           Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();
#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j]) *
               u_local[j * Nc + i].conj();  // transpose and conj
                                            // second row vector with col vector
      temp2 +=
          (src_local[1 * Nc + j] - src_local[2 * Nc + j]) * u_local[j * Nc + i].conj();  // transpose and conj
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[3 * Nc + i] += temp1;
    dst_local[1 * Nc + i] += temp2;
    dst_local[2 * Nc + i] -= temp2;
  }
}

template <>
__device__ __forceinline__ void dslashMVKernel<Z_DIM, FWD, QCU_DAGGER_NO>(Complex *u_local, Complex *src_local,
                                                                          Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j].multiply_i()) * u_local[i * Nc + j];
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j].multiply_i()) * u_local[i * Nc + j];
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[2 * Nc + i] += temp1.multiply_i();
    dst_local[1 * Nc + i] += temp2;
    dst_local[3 * Nc + i] += temp2.multiply_minus_i();
  }
}

template <>
__device__ __forceinline__ void dslashMVKernel<Z_DIM, FWD, QCU_DAGGER_YES>(Complex *u_local, Complex *src_local,
                                                                           Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j].multiply_i()) * u_local[i * Nc + j];
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j].multiply_i()) * u_local[i * Nc + j];
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[2 * Nc + i] -= temp1.multiply_i();
    dst_local[1 * Nc + i] += temp2;
    dst_local[3 * Nc + i] -= temp2.multiply_minus_i();
  }
}

template <>
__device__ __forceinline__ void dslashMVKernel<Z_DIM, BWD, QCU_DAGGER_NO>(Complex *u_local, Complex *src_local,
                                                                          Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j].multiply_i()) *
               u_local[j * Nc + i].conj();  // transpose and conj
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j].multiply_i()) *
               u_local[j * Nc + i].conj();  // transpose and conj
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[2 * Nc + i] += temp1.multiply_minus_i();
    dst_local[1 * Nc + i] += temp2;
    dst_local[3 * Nc + i] += temp2.multiply_i();
  }
}

template <>
__device__ __forceinline__ void dslashMVKernel<Z_DIM, BWD, QCU_DAGGER_YES>(Complex *u_local, Complex *src_local,
                                                                           Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j].multiply_i()) *
               u_local[j * Nc + i].conj();  // transpose and conj
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j].multiply_i()) *
               u_local[j * Nc + i].conj();  // transpose and conj
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[2 * Nc + i] -= temp1.multiply_minus_i();
    dst_local[1 * Nc + i] += temp2;
    dst_local[3 * Nc + i] -= temp2.multiply_i();
  }
}

template <>
__device__ __forceinline__ void dslashMVKernel<T_DIM, FWD, QCU_DAGGER_NO>(Complex *u_local, Complex *src_local,
                                                                          Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j]) * u_local[i * Nc + j];
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j]) * u_local[i * Nc + j];
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[2 * Nc + i] -= temp1;
    dst_local[1 * Nc + i] += temp2;
    dst_local[3 * Nc + i] -= temp2;
  }
}
template <>
__device__ __forceinline__ void dslashMVKernel<T_DIM, FWD, QCU_DAGGER_YES>(Complex *u_local, Complex *src_local,
                                                                           Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j]) * u_local[i * Nc + j];
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j]) * u_local[i * Nc + j];
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[2 * Nc + i] += temp1;
    dst_local[1 * Nc + i] += temp2;
    dst_local[3 * Nc + i] += temp2;
  }
}
template <>
__device__ __forceinline__ void dslashMVKernel<T_DIM, BWD, QCU_DAGGER_NO>(Complex *u_local, Complex *src_local,
                                                                          Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j]) * u_local[j * Nc + i].conj();  // transpose and conj
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j]) * u_local[j * Nc + i].conj();  // transpose and conj
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[2 * Nc + i] += temp1;
    dst_local[1 * Nc + i] += temp2;
    dst_local[3 * Nc + i] += temp2;
  }
}
template <>
__device__ __forceinline__ void dslashMVKernel<T_DIM, BWD, QCU_DAGGER_YES>(Complex *u_local, Complex *src_local,
                                                                           Complex *dst_local) {
  Complex temp1;
  Complex temp2;

#pragma unroll
  for (int i = 0; i < Nc; i++) {
    temp1.clear2Zero();
    temp2.clear2Zero();

#pragma unroll
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j]) * u_local[j * Nc + i].conj();  // transpose and conj
      // second row vector with col vector
      temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j]) * u_local[j * Nc + i].conj();  // transpose and conj
    }
    dst_local[0 * Nc + i] += temp1;
    dst_local[2 * Nc + i] -= temp1;
    dst_local[1 * Nc + i] += temp2;
    dst_local[3 * Nc + i] -= temp2;
  }
}