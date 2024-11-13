#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <RefSampler.h>
#include <PhaseShift.h>
#include <Selector.h>
#include <FFT.h>

// includes 
// (1) injection of the sources  
// (2) propagating wavefields in the volume [nz, ns, nw, ny, nx]
// (3) extracting at the receivers

// WEM: slowness -> data
class Propagator : public CudaOperator<complex4DReg, complex2DReg>  {
public:
  Propagator (const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range, std::shared_ptr<Arrow::RecordBatch> traces, std::shared_ptr<paramObj> par,
  complex_vector* model = nullptr, complex_vector* data = nullptr) :
  CudaOperator<complex4DReg, complex2DReg>(domain, range, model, data) {

    wfld = create_wavefield();

    // pass coords to injection later in forward or adjoint
    src_keys = {"sx", "sy", "sz", "ids"};
    inj_src = std::make_unique<Injection>(domain, wfld->getHyper(), model_vec, nullptr, grid, block);
    // pass coords to injection later in forward or adjoint
    rec_keys = {"rx", "ry", "rz", "ids"};
    inj_rec = std::make_unique<Injection>(range, wfld->getHyper(), model_vec, nullptr, grid, block);

  };

  void set_current_model(std::shared_ptr<Arrow::RecordBatch> model) {
    // pass the pointer to the new slowness model
    ref->compute(model->GetColumnByName("slowness")->data());
    // set density or impedance as well
  };
  
  // the regular forward and adjoint are not defined
  void forward(bool add, std::shared_ptr<complex2DReg>& model, std::shared_ptr<complex2DReg>& data) = delete;
  void adjoint(bool add, std::shared_ptr<complex2DReg>& model, std::shared_ptr<complex2DReg>& data) = delete;
  ///////////////////////

  void forward(bool add, const std::shared_ptr<Arrow::RecordBatch>& model, std::shared_ptr<Arrow::RecordBatch>& data) {

    if (!add) call_zero_function(data);
    
    std::shared_ptr<Arrow::Array> cx = model->GetColumnByName("sx");

    inj_src->set_coords(cx, cy, cz, ids);
    inj_src->cu_forward(false, model, wfld);

    this->cu_forward(wfld);

    inj_rec->set_coords(cx, cy, cz, ids);
    inj_rec->cu_adjoint(add, data, wfld);
  };

  void adjoint(bool add, std::shared_ptr<Arrow::RecordBatch>& model, const std::shared_ptr<Arrow::RecordBatch>& data);



  virtual ~OneWay() {
    CHECK_CUDA_ERROR(cudaFree(curr));
    CHECK_CUDA_ERROR(cudaFree(next));
  };


protected:
  std::map<std::shared<Arrow::Array>> meta;
  std::vector<std::string>> keys;
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