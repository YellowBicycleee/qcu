#pragma once

// SU(N) gauge
namespace qcu {

// USE this to store the gauge field addr
template <int _Nrhs> struct MrhsFermion {
  void *latticeFermion[_Nrhs];

  MrhsFermion() = default;
  MrhsFermion(void **latticeFermion) {
    for (int i = 0; i < _Nrhs; i++) {
      this->latticeFermion[i] = latticeFermion[i];
    }
  }
  MrhsFermion(const MrhsFermion &rhs) {
    for (int i = 0; i < _Nrhs; i++) {
      this->latticeFermion[i] = rhs.latticeFermion[i];
    }
  }
};


// for SU(N) MRHS fermion、
// WilsonDslash
// MRHS 4 DIM
template <int _Nc, int _Nrhs> __global__ wilsonDslash_suN_Mrhs4D_k(void *fermionOut, void *fermionIn) {
  MrhsFermion<_Nrhs> *rhsFermionOut = static_cast<MrhsFermion<_Nrhs> *> fermionOut;
  MrhsFermion<_Nrhs> *rhsFermionIn = static_cast<MrhsFermion<_Nrhs> *> fermionIn;
//   buff[_Nrhs * Ns * Nc]

}

} // namespace qcu
