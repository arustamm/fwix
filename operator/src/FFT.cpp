#include "FFT.h"

using namespace SEP;

cuFFT2d::cuFFT2d(const std::shared_ptr<hypercube>& domain, complex_vector* model, complex_vector* data, 
dim3 grid, dim3 block)
: CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block) {
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

  temp = make_complex_vector(domain, model_vec->_grid_, data_vec->_block_);
};

// this is on-device function
void cuFFT2d::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
  if (!add) data->zero();
  cufftExecC2C(plan, model->mat, temp->mat, CUFFT_FORWARD);
  data->add(temp);
};

// this is on-device function
void cuFFT2d::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
  if (!add) model->zero();
  cufftExecC2C(plan, data->mat, temp->mat, CUFFT_INVERSE);
  model->add(temp);
};

// this is on-device function
void cuFFT2d::cu_forward(__restrict__ complex_vector* data) {
  cufftExecC2C(plan, data->mat, data->mat, CUFFT_FORWARD);
};

// this is on-device function
void cuFFT2d::cu_adjoint(__restrict__ complex_vector* data) {
  cufftExecC2C(plan, data->mat, data->mat, CUFFT_INVERSE);
};

