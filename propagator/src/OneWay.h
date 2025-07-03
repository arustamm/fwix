#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <OneStep.h>
#include <Reflect.h>

#include <sep_reg_file.h>
#include <utils.h>
#include <ioModes.h>
#include <WavefieldPool.h>

// propagating wavefields in the volume [nz, ns, nw, ny, nx] from 0 to nz-1
class OneWay : public CudaOperator<complex4DReg, complex4DReg>  {
public:
  OneWay (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par, 
  std::shared_ptr<WavefieldPool> wfld_pool = nullptr,
  complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream),
  _slow_(slow), _param(par) {

    initialize(domain, slow->getHyper(), par, wfld_pool);
    // for now only support PSPI propagator
    prop = std::make_shared<PSPI>(domain, slow, par, model_vec, data_vec, _grid_, _block_, _stream_);

  };

  OneWay (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par, 
    std::shared_ptr<RefSampler> ref = nullptr,
    std::shared_ptr<WavefieldPool> wfld_pool = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
    CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream) {
  
      initialize(domain, slow_hyper, par, wfld_pool);
      // for now only support PSPI propagator
      prop = std::make_shared<PSPI>(domain, slow_hyper, par, ref, model_vec, data_vec, _grid_, _block_, _stream_);
    };

  std::shared_ptr<complex4DReg> get_wfld_slice(int iz) {
    // decompress and place in the 0-th buffer of the wavefield pool
    _wfld_pool->decompress_slice(iz, _compressed_wflds[iz]);
    return _wfld_pool->get_wfld_buffer(0);
  }

  void set_background_model(std::shared_ptr<complex4DReg> slow) {
    _slow_ = slow;
  }

  virtual ~OneWay() {};

  void one_step_fwd(int iz, complex_vector* __restrict__ wfld);
  void one_step_adj(int iz, complex_vector* __restrict__ wfld);

  void check_ready() { _wfld_pool->check_ready(); }
  void wait_to_finish() { _wfld_pool->wait_to_finish(); }

  size_t get_total_compressed_size() const {
    size_t total_size = 0;
    for (const auto& wfld : _compressed_wflds) {
        total_size += wfld.size();
    }
    return total_size;
  }

  double get_compression_ratio() const {
    double compressed_size = static_cast<double>(get_total_compressed_size());
    double original_total_size = static_cast<double>(getDomainSizeInBytes() * _compressed_wflds.size());
    
    if (compressed_size == 0) return 0.0;  // Avoid division by zero
    
    return original_total_size / compressed_size;
}

  std::shared_ptr<OneStep> getPropagator() {
    return prop;
  }

  std::shared_ptr<complex4DReg> getSlow() {
    return _slow_;
  }

  std::shared_ptr<paramObj> getPar() {
    return _param;
  }



protected:
  std::vector<axis> m_ax;
  // need slowness for split step propagator
  std::shared_ptr<complex4DReg> _slow_;
  std::shared_ptr<paramObj> _param;

  std::shared_ptr<WavefieldPool> _wfld_pool;
  std::vector<std::vector<char>> _compressed_wflds;

  std::shared_ptr<OneStep> prop;

private:
  void initialize(std::shared_ptr<hypercube> domain, std::shared_ptr<hypercube> slow_hyper, std::shared_ptr<paramObj> par, std::shared_ptr<WavefieldPool> wfld_pool) {
    auto ax = domain->getAxes();
    m_ax = slow_hyper->getAxes();

    if (!wfld_pool) {
      _wfld_pool = std::make_shared<WavefieldPool>(domain, par);
    } else {
      _wfld_pool = wfld_pool;
    }

    _compressed_wflds.resize(m_ax[3].n); // Resize to number of slices in z-direction
  }
};

class Downward : public OneWay {
public:
  Downward (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par,
  std::shared_ptr<WavefieldPool> wfld_pool = nullptr,
  complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneWay(domain, slow, par, wfld_pool, model, data, grid, block, stream) {};

  Downward (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par,
    std::shared_ptr<RefSampler> ref = nullptr,
    std::shared_ptr<WavefieldPool> wfld_pool = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneWay(domain, slow_hyper, par, ref, wfld_pool, model, data, grid, block, stream) {};

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
};

class Upward : public OneWay {
public:
  Upward (const std::shared_ptr<hypercube>& domain,
  std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par,
  std::shared_ptr<WavefieldPool> wfld_pool = nullptr,
  complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneWay(domain, slow, par, wfld_pool, model, data, grid, block, stream) {};

  Upward (const std::shared_ptr<hypercube>& domain,
    std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par,
    std::shared_ptr<RefSampler> ref = nullptr,
    std::shared_ptr<WavefieldPool> wfld_pool = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneWay(domain, slow_hyper, par, ref, wfld_pool, model, data, grid, block, stream) {};

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);

};