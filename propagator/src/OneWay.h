#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <RefSampler.h>
#include <PhaseShift.h>
#include <Selector.h>
#include <FFT.h>

// propagating wavefields in the volume [nz, ns, nw, ny, nx] from 0 to nz-1
class OneWay : public CudaOperator<complex5DReg, complex5DReg>  {
public:
  OneWay (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par, std::shared_ptr<RefSampler> ref, complex_vector* model = nullptr, complex_vector* data = nullptr) :
  CudaOperator<complex5DReg, complex5DReg>(domain, domain, model, data) {

    // create vector for views (does not acquire data)
    curr = model_vec->make_view();
    next = model_vec->make_view();

  };

  void cu_forward(bool add, const complex_vector* __restrict__ model, complex_vector* __restrict__ data) = 0;
	void cu_adjoint(bool add, complex_vector* __restrict__ model, const complex_vector* __restrict__ data) ;


  virtual ~OneWay() {
    curr->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(curr));
    next->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(next));
  };


protected:
  complex_vector* curr;
  complex_vector* next;

};

class Downward : public OneWay {
public:
  Downward (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par, std::shared_ptr<RefSampler> ref, 
  complex_vector* model = nullptr, complex_vector* data = nullptr) :
  OneWay(domain, slow, par, ref, model, data) {};

  void cu_forward (bool add, const complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, const complex_vector* __restrict__ data);
};

class Upward : public OneWay {
public:
  Upward (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par, std::shared_ptr<RefSampler> ref, 
  complex_vector* model = nullptr, complex_vector* data = nullptr) :
  OneWay(domain, slow, par, ref, model, data) {};

  void cu_forward (bool add, const complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, const complex_vector* __restrict__ data);
};