#include "FFT.h"

using namespace SEP;

cuFFT2d::cuFFT2d(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range, bool from_host)
: CudaOperator<complex4DReg, complex4DReg>(domain, range, from_host) {
  // create plan
  
  NX = getDomain()->getAxis(1).n;
  NY = getDomain()->getAxis(2).n;
  BATCH = getDomain()->getN123() / (NX*NY);
  SIZE = getDomain()->getN123();

  int rank = 2;
  int dims[2] = {NX, NY};

  cufftPlanMany(&plan, rank, dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, BATCH);
  // set the callback to make it orthogonal
  register_ortho_callback();
};

// this is on-device function
void cuFFT2d::cu_forward(bool add, ComplexVectorMap& model, ComplexVectorMap& data) {
  auto m_ptr = reinterpret_cast<cufftComplex*>(model_vec["host"]->mat);
  auto d_ptr = reinterpret_cast<cufftComplex*>(data_vec["host"]->mat);
  cufftExecC2C(plan, m_ptr, d_ptr, CUFFT_FORWARD);
};

// this is on-device function
void cuFFT2d::cu_adjoint(bool add, ComplexVectorMap& model, ComplexVectorMap& data) {
  auto m_ptr = reinterpret_cast<cufftComplex*>(model_vec["host"]->mat);
  auto d_ptr = reinterpret_cast<cufftComplex*>(data_vec["host"]->mat);
  cufftExecC2C(plan, d_ptr, m_ptr, CUFFT_INVERSE);
};

