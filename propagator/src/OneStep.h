#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <RefSampler.h>
#include <PhaseShift.h>
#include <Selector.h>
#include <FFT.h>
  // operator to propagate 2D wavefield ONCE in (x-y) for multiple sources and freqs (ns-nw) 
class OneStep : public CudaOperator<complex4DReg, complex4DReg>  {
public:
  OneStep (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par, std::shared_ptr<RefSampler> ref, 
  complex_vector* model = nullptr, complex_vector* data = nullptr) :
  CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data) {

    _nref_ = par->getInt("nref",1);
    _ref_ = ref;
    ps = std::make_unique<PhaseShift>(domain, slow->getHyper()->getAxis(4).d, par->getFloat("eps",0.04), model_vec, data_vec);

    _wfld_ref = make_complex_vector(domain);
    model_k = make_complex_vector(domain);

    fft2d = std::make_unique<cuFFT2d>(domain, model_vec, data_vec);
    select = std::make_unique<Selector>(domain, model_vec, data_vec);
  };

  virtual ~OneStep() {
    _wfld_ref->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(_wfld_ref));
    model_k->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(model_k));
  };

  void set_depth(int iz) {
    _iz_ = iz;
    select->set_labels(_ref_->get_ref_labels(iz));
  };
  int& get_depth() {return _iz_;};

protected:
  complex_vector* _wfld_ref;
  complex_vector* model_k;
  int _nref_, _iz_;
  float _dz_;
  std::shared_ptr<RefSampler> _ref_;
  std::unique_ptr<PhaseShift> ps;
  std::unique_ptr<cuFFT2d> fft2d;
  std::unique_ptr<Selector> select;

};

class PSPI : public OneStep {
public:
  PSPI (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par, std::shared_ptr<RefSampler> ref, 
  complex_vector* model = nullptr, complex_vector* data = nullptr) :
  OneStep(domain, slow, par, ref, model, data) {};

  void cu_forward (bool add, const complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, const complex_vector* __restrict__ data);
};

class NSPS : public OneStep {
public:
  NSPS (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par, std::shared_ptr<RefSampler> ref, 
  complex_vector* model = nullptr, complex_vector* data = nullptr) :
  OneStep(domain, slow, par, ref, model, data) {};

  void cu_forward (bool add, const complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, const complex_vector* __restrict__ data);
};