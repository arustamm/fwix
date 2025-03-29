#include <Propagator.h>

// Here I treat Propagator as a linear operator from wavelet -> data
Propagator::Propagator (const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range, std::shared_ptr<hypercube> slow_hyper,
  std::shared_ptr<complex2DReg> wavelet, 
    const std::vector<float>& sx, const std::vector<float>& sy, const std::vector<float>& sz, 
    const std::vector<int>& s_ids,
    const std::vector<float>& rx, const std::vector<float>& ry, const std::vector<float>& rz, 
    const std::vector<int>& r_ids, 
    std::shared_ptr<paramObj> par,
    complex_vector* model, complex_vector* data,
    dim3 grid, dim3 block, cudaStream_t stream) :
  CudaOperator<complex2DReg, complex2DReg>(domain, range, model, data, grid, block, stream) {

    // Here we have alredy allocated the model (source traces) and data (receiver traces) vectors on GPU

    // model_vec -> wavelet
    // data_vec -> recorded data
    int nshot = find_number_of_shots(s_ids);
    ax = slow_hyper->getAxes();
    auto wfld_hyper = std::make_shared<hypercube>(ax[0], ax[1], ax[2], nshot);

    // inj_src will allocate data_vec(wavefield) on GPU
    inj_src = std::make_unique<Injection>(wavelet->getHyper(), wfld_hyper, slow_hyper->getAxis(4).o,  slow_hyper->getAxis(4).d, sx, sy, sz, s_ids, 
    this->model_vec, nullptr, _grid_, _block_, _stream_);
    // copy wavelet to inj_src->model_vec
  CHECK_CUDA_ERROR(cudaMemcpyAsync(inj_src->model_vec->mat, wavelet->getVals(), inj_src->getDomainSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
    // in inj_rec we reuse the same data_vec (wavefield) as in inj_src and allocate a new model_vec (recorded data)
    inj_rec = std::make_unique<Injection>(range, wfld_hyper, slow_hyper->getAxis(4).o,  slow_hyper->getAxis(4).d, rx, ry, rz, r_ids, 
                                          this->data_vec, inj_src->data_vec, _grid_, _block_, _stream_);                                          
    
    ref = std::make_unique<RefSampler>(slow_hyper, par->getInt("nref"));
    down = std::make_unique<Downward>(wfld_hyper, slow_hyper, par, ref, inj_src->data_vec, inj_src->data_vec, _grid_, _block_, _stream_);
    up = std::make_unique<Upward>(wfld_hyper, slow_hyper, par, ref, inj_src->data_vec, inj_src->data_vec, _grid_, _block_, _stream_);
    // TODO: reflect to take nullptr as slowness and add set_background_model function
    reflect = std::make_unique<Reflect>(wfld_hyper, slow_hyper, inj_src->data_vec, inj_src->data_vec, _grid_, _block_, _stream_);

};

int Propagator::find_number_of_shots(const std::vector<int>& ids) {
  std::unordered_set<int> unique_ids(ids.begin(), ids.end());
  return unique_ids.size();
}

void Propagator::set_background_model(std::vector<std::shared_ptr<complex4DReg>> model) {
  down->set_background_model(model[0]);
  up->set_background_model(model[0]);
  reflect->set_background_model(model);
}

// slowness + impedance model -> to recorded data
void Propagator::nl_forward(bool add, std::vector<std::shared_ptr<complex4DReg>> model, std::shared_ptr<complex2DReg> data) {

  CHECK_CUDA_ERROR(cudaHostRegister(model[0]->getVals(), getDomainSizeInBytes(), cudaHostRegisterDefault));
  CHECK_CUDA_ERROR(cudaHostRegister(model[1]->getVals(), getDomainSizeInBytes(), cudaHostRegisterDefault));
  CHECK_CUDA_ERROR(cudaHostRegister(data->getVals(), getRangeSizeInBytes(), cudaHostRegisterDefault));

	if(!add) data->zero();

  // update model and notify all operators
  this->set_background_model(model);
  // for (batches in z)
  // down and record
  std::future<void> sample_ref;
	for (int iz=0; iz < ax[3].n-1; ++iz) {
    // sample reference slowness at current depth and return a future
    sample_ref = ref->sample_at_depth_async(model[0], iz);
    inj_src->set_depth(iz);
    inj_src->cu_forward(true, inj_src->model_vec, down->data_vec);
    inj_rec->set_depth(iz);
    inj_rec->cu_adjoint(true, inj_rec->model_vec, down->data_vec);
    // wait for the reference sampling to finish
    sample_ref.wait();
    // propagate wavefield
    down->one_step_fwd(iz, down->data_vec);
  }

  // no need to sample reference slowness again as the RefSampler already holds all the refernce velocities
  // up + reflect and record
  // TODO: I can reuse the wfld_k or _wfld_ref from the oneway operators for reflected wavefield
  // for (int iz=ax[3].n-1; iz > 0; --iz) {
  //   CHECK_CUDA_ERROR(cudaMemcpyAsync(model_k->mat, down->get_wfld()->getVals() + offset, getDomainSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
	// 	reflect->set_depth(iz);
	// 	reflect->cu_forward(true, reflected, wfld);
	// 	inj_rec->set_depth(iz);
  //   inj_rec->cu_adjoint(true, data, wfld);
  //   up->one_step_fwd(iz, wfld);
	// }

  CHECK_CUDA_ERROR(cudaMemcpyAsync(data->getVals(), data_vec->mat, getRangeSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));

		// unpin the memory
  CHECK_CUDA_ERROR(cudaHostUnregister(model[0]->getVals()));
  CHECK_CUDA_ERROR(cudaHostUnregister(model[1]->getVals()));
  CHECK_CUDA_ERROR(cudaHostUnregister(data->getVals()));
	
}


// StreamingPropagator::StreamingPropagator (const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range, 
//   std::shared_ptr<hypercube> slow_hyper, std::shared_ptr<complex2DReg>> wavelet, 
//   const std::vector<float>& cx, const std::vector<float>& cy, const std::vector<float>& cz, const std::vector<int>& ids, 
//   std::shared_ptr<paramObj> par,
//   complex_vector* model = nullptr, complex_vector* data = nullptr) :
// StreamingOperator<complex2DReg, complex2DReg, ... Args>(domain, range, model, data) {


