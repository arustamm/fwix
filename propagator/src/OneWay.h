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
    return wfld;
  }
  
  size_t get_total_compressed_size() {
    size_t total_size = 0;
    for (const auto& wfld : _compressed_wflds_) 
      total_size += wfld.second; // Add the size of each compressed buffer
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

    CHECK_CUDA_ERROR(cudaHostUnregister(wfld->getVals()));
    // wfld_file->close();
    // wfld_file->remove();
    // ZFP Cleanup: Ensure ZFP objects are freed upon destruction
    if (_zfp_stream_) zfp_stream_close(_zfp_stream_);
    if (_zfp_field_) zfp_field_free(_zfp_field_);
    // ZFP Cleanup: Free all stored compressed buffers
    for (const auto& wfld : _compressed_wflds_) 
        if(wfld.first) delete[] wfld.first; // Free the char array

    
  };

  void one_step_fwd(int iz, complex_vector* __restrict__ wfld);
  void one_step_adj(int iz, complex_vector* __restrict__ wfld);

  std::unique_ptr<OneStep> prop;

  size_t get_wfld_slice_size() {
    return static_cast<size_t>(wfld->getHyper()->getN123());
  }

  size_t get_wfld_slice_size_in_bytes() {
    return this->get_wfld_slice_size() * sizeof(std::complex<float>);
  }

protected:
  std::vector<axis> m_ax;
  std::shared_ptr<complex4DReg> wfld;
  // need slowness for split step propagator
  std::shared_ptr<complex4DReg> _slow_;

  std::shared_ptr<genericReg> wfld_file;
  std::string _filename_;

  // ZFP related members
  zfp_stream* _zfp_stream_; // ZFP stream object for compression parameters
  zfp_field* _zfp_field_;   // ZFP field object for data dimensions and type
  // This will store all compressed wavefield slices
  // Each element is a pair: {pointer to compressed data, actual size of compressed data}
  std::vector<std::pair<char*, size_t>> _compressed_wflds_;

  std::future<void> compress_slice(int iz);
  void decompress_slice(int iz);

private:
  void initialize(std::shared_ptr<hypercube> domain, std::shared_ptr<hypercube> slow_hyper, std::shared_ptr<paramObj> par) {
    auto ax = domain->getAxes();
    m_ax = slow_hyper->getAxes();
    // make a 5d wfld to store [nz, ns, nw, ny ,nx]
    auto hyper = std::make_shared<hypercube>(ax[0], ax[1], ax[2], ax[3]);
    wfld = std::make_shared<complex4DReg>(hyper);
    CHECK_CUDA_ERROR(cudaHostRegister(wfld->getVals(), hyper->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));

    // std::vector<std::string> args;
    // std::shared_ptr<ioModes> io = std::make_shared<ioModes>(args);
    // std::shared_ptr<genericIO> gen = io->getIO("SEP");

    // _filename_ = get_output_filepath(filename + "_" + generate_random_code());

    // wfld_file = gen->getReg(_filename_, SEP::usageScr, hyper->getNdim());

    // --- ZFP Compression Setup ---
    // Get a float pointer to the original complex data (interleaved real/imaginary parts)
    _zfp_stream_ = zfp_stream_open(NULL); // Allocate zfp stream
    _zfp_field_ = zfp_field_4d(NULL, zfp_type_float, 2*ax[0].n, ax[1].n, ax[2].n, ax[3].n);

    double rel_error_bound = par->getFloat("compress_error", 1E-6);
    zfp_stream_set_accuracy(_zfp_stream_, rel_error_bound);

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