#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <OneStep.h>
#include <Reflect.h>

#include <sep_reg_file.h>
#include <utils.h>
#include <ioModes.h>
#include "zfp.h"
#include <queue>

// propagating wavefields in the volume [nz, ns, nw, ny, nx] from 0 to nz-1
class OneWay : public CudaOperator<complex4DReg, complex4DReg>  {
public:
  OneWay (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par, complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream) {

    initialize(domain, slow->getHyper(), par);
    // for now only support PSPI propagator
    prop = std::make_unique<PSPI>(domain, slow, par, model_vec, data_vec, _grid_, _block_, _stream_);

  };

  OneWay (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par, 
    std::shared_ptr<RefSampler> ref = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
    CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream) {
  
      initialize(domain, slow_hyper, par);
      // for now only support PSPI propagator
      prop = std::make_unique<PSPI>(domain, slow_hyper, par, ref, model_vec, data_vec, _grid_, _block_, _stream_);
    };

  std::shared_ptr<complex4DReg> get_wfld_slice(int iz) {
    // decompress
    this->decompress_slice(iz);
    return _wfld_pool[0];
  }
  
  size_t get_total_compressed_size() {
    size_t total_size = 0;
    for (const auto& wfld : _compressed_wflds_) 
      total_size += wfld.size(); // Add the size of each compressed buffer
    return total_size;
  }

  double get_compression_ratio() {
  
    double compressed_size = static_cast<double>(this->get_total_compressed_size());
    double total_size = static_cast<double>(this->getDomainSizeInBytes() * m_ax[3].n);
    double ratio = total_size / compressed_size;

    return ratio;
  }

  void set_background_model(std::shared_ptr<complex4DReg> slow) {
    _slow_ = slow;
  }

  virtual ~OneWay() {
    while (!_compression_futures.empty()) {
        _compression_futures.front().wait();
        _compression_futures.pop();
    }

    for (const auto& wfld_ptr : _wfld_pool) {
        if (wfld_ptr && wfld_ptr->getVals()) {
            // This is the counterpart to cudaHostRegister
            CHECK_CUDA_ERROR(cudaHostUnregister(wfld_ptr->getVals()));
        }
    }

    for (auto event : _events_pool) {
        if (event) {
            CHECK_CUDA_ERROR(cudaEventDestroy(event));
        }
    }

    for (size_t i = 0; i < _zfp_stream_pool.size(); ++i) {
        if (_zfp_stream_pool[i]) {
            zfp_stream_close(_zfp_stream_pool[i]);
        }
        if (_zfp_field_pool[i]) {
            zfp_field_free(_zfp_field_pool[i]);
        }
    }

};

  void one_step_fwd(int iz, complex_vector* __restrict__ wfld);
  void one_step_adj(int iz, complex_vector* __restrict__ wfld);

  std::unique_ptr<OneStep> prop;

  size_t get_wfld_slice_size() {
    return static_cast<size_t>(_wfld_pool[0]->getHyper()->getN123());
  }

  size_t get_wfld_slice_size_in_bytes() {
    return this->get_wfld_slice_size() * sizeof(std::complex<float>);
  }

  void wait_for_pipeline();
  void check_pipeline();

protected:
  std::vector<axis> m_ax;
  std::vector<std::shared_ptr<complex4DReg>> _wfld_pool;
  std::vector<cudaEvent_t> _events_pool;
  // need slowness for split step propagator
  std::shared_ptr<complex4DReg> _slow_;

  std::shared_ptr<genericReg> wfld_file;
  std::string _filename_;
  int _nwflds_to_store;

   // A queue to hold futures for the compression tasks
  std::queue<std::future<void>> _compression_futures;

  // ZFP related members
  std::vector<zfp_stream*> _zfp_stream_pool;
  std::vector<zfp_field*> _zfp_field_pool;
  // This will store all compressed wavefield slices
  // Each element is a pair: {pointer to compressed data, actual size of compressed data}
  std::vector<std::vector<char>> _compressed_wflds_;

  std::future<void> compress_slice(int iz, int pool_idx, cudaEvent_t event);
  void decompress_slice(int iz);


private:
  void initialize(std::shared_ptr<hypercube> domain, std::shared_ptr<hypercube> slow_hyper, std::shared_ptr<paramObj> par) {
    auto ax = domain->getAxes();
    m_ax = slow_hyper->getAxes();

    _nwflds_to_store = par->getInt("wflds_to_store", 3); // Let's use a depth of 3

    // Create the buffer and event pool
    _wfld_pool.resize(_nwflds_to_store);
    _events_pool.resize(_nwflds_to_store);
    for (int i = 0; i < _nwflds_to_store; ++i) {
        auto hyper = std::make_shared<hypercube>(ax[0], ax[1], ax[2], ax[3]);
        _wfld_pool[i] = std::make_shared<complex4DReg>(hyper);
        CHECK_CUDA_ERROR(cudaHostRegister(_wfld_pool[i]->getVals(), hyper->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));
        CHECK_CUDA_ERROR(cudaEventCreate(&_events_pool[i]));
    }

    // std::vector<std::string> args;
    // std::shared_ptr<ioModes> io = std::make_shared<ioModes>(args);
    // std::shared_ptr<genericIO> gen = io->getIO("SEP");

    // _filename_ = get_output_filepath(filename + "_" + generate_random_code());

    // wfld_file = gen->getReg(_filename_, SEP::usageScr, hyper->getNdim());

    // --- ZFP Compression Setup ---
    // Get a float pointer to the original complex data (interleaved real/imaginary parts)
    _zfp_stream_pool.resize(_nwflds_to_store);
    _zfp_field_pool.resize(_nwflds_to_store);
    double rel_error_bound = par->getFloat("compress_error", 1E-6);

    for (int i = 0; i < _nwflds_to_store; ++i) {
        _zfp_stream_pool[i] = zfp_stream_open(NULL);
        zfp_stream_set_accuracy(_zfp_stream_pool[i], rel_error_bound);
        if (!zfp_stream_set_execution(_zfp_stream_pool[i], zfp_exec_omp)) {
          std::cerr << "Warning: Failed to set ZFP OpenMP execution policy." << std::endl;
          // Handle error or fall back to serial
        }
        // Note: The data pointer is null, it will be set just-in-time
        _zfp_field_pool[i] = zfp_field_4d(nullptr, zfp_type_float, 2*ax[0].n, ax[1].n, ax[2].n, ax[3].n);
    }

    _compressed_wflds_.resize(m_ax[3].n); // Resize to number of slices in z-direction
  }
};

class Downward : public OneWay {
public:
  Downward (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par,
  complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneWay(domain, slow, par, model, data, grid, block, stream) {};

  Downward (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par,
    std::shared_ptr<RefSampler> ref = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneWay(domain, slow_hyper, par, ref, model, data, grid, block, stream) {};

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
};

class Upward : public OneWay {
public:
  Upward (const std::shared_ptr<hypercube>& domain,
  std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par,
  complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneWay(domain, slow, par, model, data, grid, block, stream) {};

  Upward (const std::shared_ptr<hypercube>& domain,
    std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par,
    std::shared_ptr<RefSampler> ref = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneWay(domain, slow_hyper, par, ref, model, data, grid, block, stream) {};

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);

};