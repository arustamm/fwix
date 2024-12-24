#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <OneStep.h>

// propagating wavefields in the volume [nz, ns, nw, ny, nx] from 0 to nz-1
class OneWay : public CudaOperator<complex5DReg, complex5DReg>  {
public:
  OneWay (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par, complex_vector* model = nullptr, complex_vector* data = nullptr) :
  CudaOperator<complex5DReg, complex5DReg>(domain, domain, model, data) {

    // create vector for views (does not acquire data)
    curr = model_vec->make_view();
    next = model_vec->make_view();

    ax = domain->getAxes();
    // hyper = [ns, nw, nx ,ny]
    auto hyper = std::make_shared<hypercube>(ax[0], ax[1], ax[2], ax[3]);
    // for now only support PSPI propagator
    _grid_ = {128, 128, 8};
    _block_ = {16, 16, 2};
    curr->set_grid_block(_grid_, _block_);
    next->set_grid_block(_grid_, _block_);
    prop = std::make_unique<PSPI>(hyper, slow, par, curr, next, _grid_, _block_);

  };

  virtual ~OneWay() {
    curr->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(curr));
    next->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(next));
  };

  void set_wfld(complex_vector* wfld) {

  };

  void save_wfld(complex_vector* wfld) {};


protected:
  complex_vector* curr;
  complex_vector* next;
  std::unique_ptr<OneStep> prop;
  std::vector<axis> ax;
};

class Downward : public OneWay {
public:
  Downward (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par,
  complex_vector* model = nullptr, complex_vector* data = nullptr) :
  OneWay(domain, slow, par, model, data) {};

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
};

class Upward : public OneWay {
public:
  Upward (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par,
  complex_vector* model = nullptr, complex_vector* data = nullptr) :
  OneWay(domain, slow, par, model, data) {};

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
};