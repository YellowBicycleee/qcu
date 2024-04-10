#include <cassert>

#include "solver/qcu_cg.h"

#define PRINT_RATIO

BEGIN_NAMESPACE(qcu)
static inline bool ifConverge(double target, double rsdNorm, double bNorm) { return rsdNorm < target * bNorm; }

// WILSON
void CGDslashMV_Odd::operator()(_genvector result, _genvector src, cudaStream_t stream) {
  DslashParam *dslashParam = dslash->dslashParam_;
  int Lx = dslashParam->Lx;
  int Ly = dslashParam->Ly;
  int Lz = dslashParam->Lz;
  int Lt = dslashParam->Lt;
  double kappa = dslashParam->kappa;
  int vol = Lx * Ly * Lz * Lt;
  int vectorLength = vol * Ns * Nc / 2;
  void *tempFermionIn1 = dslashParam->tempFermionIn1;
  void *tempFermionIn2 = dslashParam->tempFermionIn2;
  cudaStream_t stream1 = dslashParam->stream1;
  cudaStream_t stream2 = dslashParam->stream2;
  Complex minusKappaSquare = Complex(-kappa * kappa, 0.0);
  QcuSaxpy saxpyFunc;

  // if (dslashType_ == DSLASH_WILSON) {

  // no-dagger
  dslashParam->parity = EVEN_PARITY;
  dslashParam->daggerFlag = QCU_DAGGER_NO;
  dslashParam->fermionIn = src;
  dslashParam->fermionOut = tempFermionIn1;

  dslash->preApply();
  dslash->apply();
  dslash->postApply(); // tempFermionIn1 = D_{eo} x_{o}

  dslashParam->parity = ODD_PARITY;
  dslashParam->daggerFlag = QCU_DAGGER_NO;
  dslashParam->fermionIn = tempFermionIn1;
  dslashParam->fermionOut = tempFermionIn2;

  dslash->preApply();
  dslash->apply();
  dslash->postApply(); // tempFermionIn2 = D_{oe} D_{eo} x_{o}

  // 注意：tempFermion2的结果要保留到最后
  saxpyFunc(tempFermionIn2, minusKappaSquare, tempFermionIn2, src, vectorLength, stream1);

  // dagger
  dslashParam->parity = EVEN_PARITY;
  dslashParam->daggerFlag = QCU_DAGGER_YES;
  dslashParam->fermionIn = tempFermionIn2;
  dslashParam->fermionOut = tempFermionIn1;

  dslash->preApply();
  dslash->apply();
  dslash->postApply(); // tempFermionIn1 = D^{\dagger}_{eo} tempFermionIn2

  dslashParam->parity = ODD_PARITY;
  dslashParam->daggerFlag = QCU_DAGGER_YES;
  dslashParam->fermionIn = tempFermionIn1;
  dslashParam->fermionOut = result;

  dslash->preApply();
  dslash->apply();
  dslash->postApply(); // result = D^{\dagger}_{oe} D^{\dagger}_{eo} tempFermionIn2

  // result = tempFermionIn2 - kappa^2 * result
  saxpyFunc(result, minusKappaSquare, result, tempFermionIn2, vectorLength, stream1);
  // } else {
  //   assert(0);
  // }
}

void QcuCG::allocateTempVectors() {
  // allocate device memory
  int Lx = cgParam_->Lx;
  int Ly = cgParam_->Ly;
  int Lz = cgParam_->Lz;
  int Lt = cgParam_->Lt;

  int vol = Lx * Ly * Lz * Lt;
  int vectorLength = vol * Ns * Nc / 2; // /2because of even-odd preconditioning

  CHECK_CUDA(cudaMalloc(&newEvenB_, vectorLength * sizeof(double) * 2));
  CHECK_CUDA(cudaMalloc(&newOddB_, vectorLength * sizeof(double) * 2));
  CHECK_CUDA(cudaMalloc(&residual_, vectorLength * sizeof(double) * 2));
  CHECK_CUDA(cudaMalloc(&pVec_, vectorLength * sizeof(double) * 2));
  CHECK_CUDA(cudaMalloc(&tmp1_, vectorLength * sizeof(double) * 2));
  CHECK_CUDA(cudaMalloc(&tmp2_, vectorLength * sizeof(double) * 2));
  CHECK_CUDA(cudaMalloc(&tmp3_, vectorLength * sizeof(double) * 2));
  CHECK_CUDA(cudaMalloc(&tmp4_, vectorLength * sizeof(double) * 2));
  CHECK_CUDA(cudaMalloc(&tmp5_, vectorLength * sizeof(double) * 2));
  CHECK_CUDA(cudaMalloc(&dslashTempVec1_, vectorLength * sizeof(double) * 2));
  CHECK_CUDA(cudaMalloc(&dslashTempVec2_, vectorLength * sizeof(double) * 2));
}

void QcuCG::freeTempVectors() {
  CHECK_CUDA(cudaFree(newEvenB_));
  newEvenB_ = nullptr;
  CHECK_CUDA(cudaFree(newOddB_));
  newOddB_ = nullptr;
  CHECK_CUDA(cudaFree(residual_));
  residual_ = nullptr;
  CHECK_CUDA(cudaFree(pVec_));
  pVec_ = nullptr;
  CHECK_CUDA(cudaFree(tmp1_));
  tmp1_ = nullptr;
  CHECK_CUDA(cudaFree(tmp2_));
  tmp2_ = nullptr;
  CHECK_CUDA(cudaFree(tmp3_));
  tmp3_ = nullptr;
  CHECK_CUDA(cudaFree(tmp4_));
  tmp4_ = nullptr;
  CHECK_CUDA(cudaFree(tmp5_));
  tmp5_ = nullptr;
  CHECK_CUDA(cudaFree(dslashTempVec1_));
  dslashTempVec1_ = nullptr;
  CHECK_CUDA(cudaFree(dslashTempVec2_));
  dslashTempVec2_ = nullptr;
}

bool QcuCG::odd_cg(void *resX, void *newOddB) {
  bool res = false;
  int Lx = cgParam_->Lx;
  int Ly = cgParam_->Ly;
  int Lz = cgParam_->Lz;
  int Lt = cgParam_->Lt;

  int vol = Lx * Ly * Lz * Lt;
  int vectorLength = vol * Ns * Nc / 2;
  Complex reg1;
  Complex reg2;
  Complex alpha;
  Complex beta;
  double rsdNorm = 0.0; // 残差 norm
  double bNorm = 0.0;   // b的范数
                        // 终止条件：rsdNorm  < rsdTarget * bNorm

  // calc norm of input b vector
  norm2_(tmp1_, tmp2_, newOddB, vectorLength, cgParam_->stream1);
  CHECK_CUDA(cudaMemcpyAsync(&bNorm, tmp1_, sizeof(double), cudaMemcpyDeviceToHost, cgParam_->stream1));

  // x = 0
  CHECK_CUDA(cudaMemset(resX, 0, vectorLength * sizeof(double) * 2));

  // r = b - Ax, because of x = 0, so r = b
  complexCopy_(residual_, newOddB, vectorLength, cgParam_->stream2);

  // p = r
  complexCopy_(pVec_, residual_, vectorLength, cgParam_->stream2);

  // calc r norm, donnot use tmp1_ tmp2_ because tmp1 and tmp2 are used in stream1
  norm2_(tmp3_, tmp4_, residual_, vectorLength, cgParam_->stream2);
  CHECK_CUDA(cudaMemcpyAsync(&rsdNorm, tmp3_, sizeof(double), cudaMemcpyDeviceToHost, cgParam_->stream2));

  // sync
  CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
  CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream2));
  // now, tmp1_ tmp2_ tmp3_ tmp4_ are not used

  while (numIterations_ < maxIterations_) {
    numIterations_++;
    // A * p
    cgIterMV_Odd_(tmp1_, pVec_); // tmp1_ = A * pVec_
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream2));

    // alpha = r^T * r / p^T * A * p
    // reg1 = r^T * r
    // reg2 = p^T * A * p
    innerProd_(tmp2_, tmp4_, residual_, residual_, vectorLength, cgParam_->stream1);
    CHECK_CUDA(cudaMemcpyAsync(&reg1, tmp2_, sizeof(Complex), cudaMemcpyDeviceToHost, cgParam_->stream1));

    innerProd_(tmp3_, tmp5_, pVec_, tmp1_, vectorLength, cgParam_->stream2);
    CHECK_CUDA(cudaMemcpyAsync(&reg2, tmp3_, sizeof(Complex), cudaMemcpyDeviceToHost, cgParam_->stream2));

    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream2));
    alpha = reg1 / reg2;
    // tmp2_ tmp3_ tmp4_ tmp5_ are not used anymore

    // x = x + alpha * p
    saxpy_(resX, alpha, pVec_, resX, vectorLength, cgParam_->stream1);
    // r' = r - alpha * A * p
    saxpy_(tmp2_, -alpha, tmp1_, residual_, vectorLength, cgParam_->stream2); // tmp2 = r'
    // calc r norm
    norm2_(tmp3_, tmp4_, residual_, vectorLength, cgParam_->stream2);
    CHECK_CUDA(cudaMemcpyAsync(&rsdNorm, tmp3_, sizeof(double), cudaMemcpyDeviceToHost, cgParam_->stream2));
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream2));

#ifdef PRINT_RATIO
    if (numIterations_ % 10 == 0) {
      printf("odd cg iteration %d, rsdNorm = %e, bNorm = %e, ratio = %e\n", numIterations_, rsdNorm, bNorm,
             rsdNorm / bNorm);
    }
#endif
    // if converge then return x
    if (ifConverge(rsdTarget_, rsdNorm, bNorm)) {
      // printf("odd x converged in %d iterations\n", numIterations_);
      return true;
    }

    // beta = r'^T * r' / r^T * r
    // we have reg1 = r^T * r already
    // then we only have to calc reg2 = r'^T * r', r' stored in tmp2_
    // then reg2 = r'^T * r'
    innerProd_(tmp3_, tmp4_, tmp2_, tmp2_, vectorLength, cgParam_->stream1);
    CHECK_CUDA(cudaMemcpyAsync(&reg2, tmp3_, sizeof(Complex), cudaMemcpyDeviceToHost, cgParam_->stream1));
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
    beta = reg2 / reg1;

    // p = r' + beta * p
    saxpy_(pVec_, beta, pVec_, tmp2_, vectorLength, cgParam_->stream1);
    // r = r'
    complexCopy_(residual_, tmp2_, vectorLength, cgParam_->stream2);

    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream2));
  }
  return res;
}

// wilson dslash cg: 1 iter, get new b as x
// evenB is inputEvenB rather than newEvenB
bool QcuCG::even_cg(void *resEvenX, void *evenB) {
  numIterations_++;
  generateEvenB(resEvenX, evenB); // resEvenX is newEvenB
  return true;
}
// TODO
// there, qcuInvert inputs full fermion field, and outputs full fermion field
// inputB = cgParam_->fermionInB
// resX = cgParam_->fermionOutX
void QcuCG::qcuInvert() {
  void *resX = cgParam_->fermionOutX;
  void *inputb = cgParam_->fermionInB;
  bool res = false;
  generateOddB(newOddB_, tmp1_, tmp2_); // first, generate new oddB

  numIterations_ = 0; // 迭代次数重置
  res = odd_cg(oddX_, newOddB_);
  if (!res) {
    printf("odd cg failed\n");
    return;
  }
  res = even_cg(evenX_, evenB_);
  printf("CG inverter succeed in %d iterations\n", numIterations_);
}

// Comment: (A_{oo} + \kappa^2 D_{oe} A_{ee}^{-1} D_{eo}) x_o = \kappa D_{oe}A_{ee}^{-1}b_e + b_o
// then newOddB = \kappa D_{oe}A_{ee}^{-1}b_e + b_o
// in wilson dslash , A is I, then newOddB = \kappa D_{oe}b_e + b_o
void QcuCG::generateOddB(void *new_b, void *tempVec1, void *tempVec2) {
  // in QcuCG, we have even_fermion_in and odd_fermion_in
  int Lx = cgParam_->Lx;
  int Ly = cgParam_->Ly;
  int Lz = cgParam_->Lz;
  int Lt = cgParam_->Lt;

  int vol = Lx * Ly * Lz * Lt;
  int vectorLength = vol * Ns * Nc / 2;

  Complex kappa(cgParam_->kappa, 0);
  Complex minusKappaSquare(-cgParam_->kappa * cgParam_->kappa, 0);
  Complex minusOne(-1, 0);
  DslashParam *dslashParam = dslash_->dslashParam_;

  dslashParam->parity = ODD_PARITY;
  QCU_DAGGER_FLAG daggerFlag = QCU_DAGGER_NO;

  // tempVec1 = D_{oe}b_e
  singleDslash_(tempVec1, evenB_);
  CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
  CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream2));
  // tempVec1 = right_b = \kappa D_{oe}b_e + b_o
  saxpy_(tempVec1, kappa, tempVec1, oddB_, vectorLength, cgParam_->stream1);

  CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));

  if (dslashType_ == DSLASH_WILSON) {
    // dagger
    dslashParam->parity = EVEN_PARITY;
    dslashParam->daggerFlag = QCU_DAGGER_YES;
    singleDslash_(tempVec2, tempVec1); // tempVec2 = D^{\dagger}_{oe} right_b
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream2));

    dslashParam->parity = ODD_PARITY;
    dslashParam->daggerFlag = QCU_DAGGER_YES;
    singleDslash_(new_b, tempVec2); // new_b = D^{\dagger}_{oe} D^{\dagger}_{eo} right_b
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream2));

    // new_b = tempVec1 - kappa ^ 2 * new_b
    //       = right_b - kappa ^ 2 * D^{\dagger}_{oe} D^{\dagger}_{eo} right_b
    //       = (I - kappa ^ 2 * D^{\dagger}_{oe} D^{\dagger}_{eo}) right_b
    saxpy_(new_b, minusKappaSquare, new_b, tempVec1, vectorLength, cgParam_->stream1);
    CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
  } else {
    assert(0);
  }
}

// Comment: A_{ee} x_e  - \kappa D_{eo} x_o = evenB,
// in wilson Dslash, A is I, then
// x_e - \kappa D_{eo} x_o = evenB, aka x_e = \kappa D_{eo} x_o + evenB
// new_b = x_e = \kappa D_{eo} x_o + evenB
void QcuCG::generateEvenB(void *newEvenB, void *evenB) {
  int Lx = cgParam_->Lx;
  int Ly = cgParam_->Ly;
  int Lz = cgParam_->Lz;
  int Lt = cgParam_->Lt;

  int vol = Lx * Ly * Lz * Lt;
  int vectorLength = vol * Ns * Nc / 2;

  Complex kappa(cgParam_->kappa, 0);
  DslashParam *dslashParam = dslash_->dslashParam_;

  dslashParam->parity = EVEN_PARITY;
  dslashParam->daggerFlag = QCU_DAGGER_NO;
  singleDslash_(newEvenB, oddX_); // newEvenB = D_{eo} x_o
  CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
  CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream2));

  // newEvenB = \kappa D_{eo} x_o + evenB
  saxpy_(newEvenB, kappa, newEvenB, evenB, vectorLength, cgParam_->stream1);
  CHECK_CUDA(cudaStreamSynchronize(cgParam_->stream1));
}

END_NAMESPACE(qcu)