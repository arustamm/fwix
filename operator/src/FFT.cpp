#include "FFT.h"

using namespace SEP;

cuFFT2d::cuFFT2d(const std::shared_ptr<hypercube>& domain, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data)
: CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data) {
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

  temp = make_complex_vector_map(domain, (*model_vec)["host"]->_grid_, (*data_vec)["host"]->_block_);
};

// this is on-device function
void cuFFT2d::cu_forward(bool add, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data) {
  if (!add) (*data)["host"]->zero();
  cufftExecC2C(plan, (*model)["host"]->mat, (*temp)["host"]->mat, CUFFT_FORWARD);
  (*data)["host"]->add((*temp)["host"]);
};

// this is on-device function
void cuFFT2d::cu_adjoint(bool add, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data) {
  if (!add) (*model)["host"]->zero();
  cufftExecC2C(plan, (*data)["host"]->mat, (*temp)["host"]->mat, CUFFT_INVERSE);
  (*model)["host"]->add((*temp)["host"]);
};

// this is on-device function
void cuFFT2d::cu_forward(std::shared_ptr<ComplexVectorMap> data) {
  cufftExecC2C(plan, (*data)["host"]->mat, (*data)["host"]->mat, CUFFT_FORWARD);
};

// this is on-device function
void cuFFT2d::cu_adjoint(std::shared_ptr<ComplexVectorMap> data) {
  cufftExecC2C(plan, (*data)["host"]->mat, (*data)["host"]->mat, CUFFT_INVERSE);
};

