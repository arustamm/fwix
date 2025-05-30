#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <OneStep.h>
#include <Reflect.h>

#include <sep_reg_file.h>
#include <utils.h>
#include <ioModes.h>

// propagating wavefields in the volume [nz, ns, nw, ny, nx] from 0 to nz-1
class OneWay : public CudaOperator<complex4DReg, complex4DReg>  {
public:
  OneWay (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par, complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0,
  std ::string filename = "oneway") :
  CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream) {

    initialize(domain, slow->getHyper(), filename);
    // for now only support PSPI propagator
    prop = std::make_unique<PSPI>(domain, slow, par, model_vec, data_vec, _grid_, _block_, _stream_);

  };

  OneWay (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par, 
    std::shared_ptr<RefSampler> ref = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0,
    std ::string filename = "oneway") :
    CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream) {
  
      initialize(domain, slow_hyper, filename);
      // for now only support PSPI propagator
      prop = std::make_unique<PSPI>(domain, slow_hyper, par, ref, model_vec, data_vec, _grid_, _block_, _stream_);
    };

  std::shared_ptr<complex5DReg> get_wfld() {
    return wfld;
  }

  void set_background_model(std::shared_ptr<complex4DReg> slow) {
    _slow_ = slow;
  }

  virtual ~OneWay() {
    CHECK_CUDA_ERROR(cudaHostUnregister(wfld->getVals()));
    // wfld_file->close();
    // wfld_file->remove();
  };

  void one_step_fwd(int iz, complex_vector* __restrict__ wfld);
  void one_step_adj(int iz, complex_vector* __restrict__ wfld);

  std::unique_ptr<OneStep> prop;

  size_t get_wfld_slice_size() {
    return static_cast<size_t>(wfld->getHyper()->getN123() / m_ax[3].n);
  }

  size_t get_wfld_slice_size_in_bytes() {
    return this->get_wfld_slice_size() * sizeof(std::complex<float>);
  }

protected:
  std::vector<axis> m_ax;
  std::shared_ptr<complex5DReg> wfld;
  // need slowness for split step propagator
  std::shared_ptr<complex4DReg> _slow_;

  std::shared_ptr<genericReg> wfld_file;
  std::string _filename_;

private:
  void initialize(std::shared_ptr<hypercube> domain, std::shared_ptr<hypercube> slow_hyper, std::string filename) {
    auto ax = domain->getAxes();
    m_ax = slow_hyper->getAxes();
    // make a 5d wfld to store [nz, ns, nw, ny ,nx]
    auto hyper = std::make_shared<hypercube>(ax[0], ax[1], ax[2], ax[3], m_ax[3]);
    wfld = std::make_shared<complex5DReg>(hyper);
    CHECK_CUDA_ERROR(cudaHostRegister(wfld->getVals(), hyper->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));

    // std::vector<std::string> args;
    // std::shared_ptr<ioModes> io = std::make_shared<ioModes>(args);
    // std::shared_ptr<genericIO> gen = io->getIO("SEP");

    // _filename_ = get_output_filepath(filename + "_" + generate_random_code());

    // wfld_file = gen->getReg(_filename_, SEP::usageScr, hyper->getNdim());
  }
};

class Downward : public OneWay {
public:
  Downward (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par,
  complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0,
  std::string filename = "down") :
  OneWay(domain, slow, par, model, data, grid, block, stream, filename) {};

  Downward (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par,
    std::shared_ptr<RefSampler> ref = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0, 
    std::string filename = "down") :
  OneWay(domain, slow_hyper, par, ref, model, data, grid, block, stream, filename) {};

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
};

class Upward : public OneWay {
public:
  Upward (const std::shared_ptr<hypercube>& domain,
  std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par,
  complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0,
  std::string filename = "up") :
  OneWay(domain, slow, par, model, data, grid, block, stream, filename) {};

  Upward (const std::shared_ptr<hypercube>& domain,
    std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par,
    std::shared_ptr<RefSampler> ref = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0,
    std::string filename = "up") :
  OneWay(domain, slow_hyper, par, ref, model, data, grid, block, stream, filename) {};

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);

};