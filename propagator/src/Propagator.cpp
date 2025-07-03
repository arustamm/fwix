#include <Propagator.h>
#include <WavefieldPool.h>
#include <tbb/tbb.h>
#include <tbb/parallel_pipeline.h>
// Here I treat Propagator as a linear operator from wavelet -> data
Propagator::Propagator (
  const std::shared_ptr<hypercube>& domain, 
  const std::shared_ptr<hypercube>& range, 
  std::shared_ptr<hypercube> slow_hyper,
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
  _slow_hyper = slow_hyper;
  // model_vec -> wavelet
  // data_vec -> recorded data

  // Find the minimum ID value from both source and receiver IDs using std::min_element
  int min_id = *std::min_element(s_ids.begin(), s_ids.end());

  // Create new vectors with shifted IDs using std::transform
  std::vector<int> shifted_s_ids(s_ids.size());
  std::transform(s_ids.begin(), s_ids.end(), shifted_s_ids.begin(), 
                [min_id](int id) { return id - min_id; });
  
  std::vector<int> shifted_r_ids(r_ids.size());
  std::transform(r_ids.begin(), r_ids.end(), shifted_r_ids.begin(), 
                [min_id](int id) { return id - min_id; });
  
  // Find the maximum ID to calculate nshot
  int max_shifted_id = *std::max_element(shifted_s_ids.begin(), shifted_s_ids.end());
  int nshot = max_shifted_id + 1; // since IDs start at 0

  ax = slow_hyper->getAxes();
  // take care of padding 
  ax[0].n += par->getInt("padx", 0);
  ax[1].n += par->getInt("pady", 0);

  auto wfld_hyper = std::make_shared<hypercube>(ax[0], ax[1], ax[2], nshot);

  // inj_src will allocate data_vec(wavefield) on GPU
  inj_src = std::make_unique<Injection>(wavelet->getHyper(), wfld_hyper, slow_hyper->getAxis(4).o,  slow_hyper->getAxis(4).d, sx, sy, sz, shifted_s_ids, 
  this->model_vec, nullptr, _grid_, _block_, _stream_);
  // copy wavelet to inj_src->model_vec
  CHECK_CUDA_ERROR(cudaHostRegister(wavelet->getVals(), inj_src->getDomainSizeInBytes(), cudaHostRegisterDefault));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(inj_src->model_vec->mat, wavelet->getVals(), inj_src->getDomainSizeInBytes(), cudaMemcpyHostToDevice, _stream_));

  // in inj_rec we reuse the same data_vec (wavefield) as in inj_src and allocate a new model_vec (recorded data)
  inj_rec = std::make_unique<Injection>(range, wfld_hyper, slow_hyper->getAxis(4).o,  slow_hyper->getAxis(4).d, rx, ry, rz, shifted_r_ids, 
                                        this->data_vec, inj_src->data_vec, _grid_, _block_, _stream_);                                          
  
  ref = std::make_unique<RefSampler>(slow_hyper, par);
  auto wfld_pool = std::make_shared<WavefieldPool>(wfld_hyper, par);
  // TODO: dont have to allocate temp wavefields twice for up and down, can reuse instead
  down = std::make_unique<Downward>(wfld_hyper, slow_hyper, par, ref, wfld_pool, inj_src->data_vec, inj_src->data_vec, _grid_, _block_, _stream_);
  up = std::make_unique<Upward>(wfld_hyper, slow_hyper, par, ref, wfld_pool, inj_src->data_vec, inj_src->data_vec, _grid_, _block_, _stream_);
  reflect = std::make_unique<Reflect>(wfld_hyper, slow_hyper, inj_src->data_vec, inj_src->data_vec, _grid_, _block_, _stream_);

  look_ahead = par->getInt("look_ahead", 1);

  CHECK_CUDA_ERROR(cudaHostUnregister(wavelet->getVals()));
};

void Propagator::set_background_model(std::vector<std::shared_ptr<complex4DReg>> model) {
  down->set_background_model(model[0]);
  up->set_background_model(model[0]);
  reflect->set_background_model(model);
}

// slowness + impedance model -> to recorded data
void Propagator::forward(bool add, std::vector<std::shared_ptr<complex4DReg>> model, std::shared_ptr<complex2DReg> data) {

  bool same = model[0]->getHyper()->checkSame(_slow_hyper);
  if (!same) 
    throw std::runtime_error("Error: model hypercube does not match the slow_hyper");
  if (!data->getHyper()->checkSame(this->getRange())) 
    throw std::runtime_error("Error: data hypercube does not match the range");

  CHECK_CUDA_ERROR(cudaHostRegister(model[0]->getVals(), model[0]->getHyper()->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));
  CHECK_CUDA_ERROR(cudaHostRegister(model[1]->getVals(), model[1]->getHyper()->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));
  CHECK_CUDA_ERROR(cudaHostRegister(data->getVals(), getRangeSizeInBytes(), cudaHostRegisterDefault));

  // always zero out the internal data_vec that records the data
	this->data_vec->zero();
  if(!add) data->zero();

  // update model and notify all operators
  this->set_background_model(model);
  // for (batches in z)

  int max_depth = ax[3].n;

 // Start with the first depth
 std::future<void> current_future = ref->sample_at_depth_async(model[0], 0);
  
 // Create a queue of futures for prefetched depths
 std::queue<std::future<void>> future_queue;
 
 // Prefetch initial depths based on look_ahead parameter
 for (int i = 1; i < std::min(look_ahead + 1, max_depth); i++) 
   future_queue.push(ref->sample_at_depth_async(model[0], i));

 // Process all depths
 for (int iz = 0; iz < max_depth; iz++) {
   // Start sampling the next depth that needs sampling
  int next_depth = iz + look_ahead + 1;
  if (next_depth < max_depth)   // Only schedule if within bounds
      future_queue.push(ref->sample_at_depth_async(model[0], next_depth));

   // Process current depth steps that don't need the sampled data
   inj_src->set_depth(iz);
   inj_src->cu_forward(true, inj_src->model_vec, down->data_vec);
   
   inj_rec->set_depth(iz);
   inj_rec->cu_adjoint(true, this->data_vec, down->data_vec);
   
   // Wait for current sampling to complete
   current_future.wait();
   
   // Propagate wavefield
	 down->check_ready();
   down->one_step_fwd(iz, down->data_vec);
   
   // Update current_future for the next iteration
   if (!future_queue.empty()) {
     current_future = std::move(future_queue.front());
     future_queue.pop();
   }
 }
 	
 	down->wait_to_finish();

  up->data_vec->zero();
  // no need to sample reference slowness again as the RefSampler already holds all the refernce velocities
  // up + reflect and record
  auto down_wfld = down->getPropagator()->get_ref_wfld();
  for (int iz=ax[3].n-1; iz >= 0; --iz) {
    down_wfld->zero();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(down_wfld->mat, down->get_wfld_slice(iz)->getVals(), down->getDomainSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
		CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_));

    reflect->set_depth(iz);
		reflect->cu_forward(false, down_wfld, up->data_vec);

		inj_rec->set_depth(iz);
    inj_rec->cu_adjoint(true, this->data_vec, up->data_vec);

		up->check_ready();
    up->one_step_fwd(iz, up->data_vec);
	}
	up->wait_to_finish();

  CHECK_CUDA_ERROR(cudaMemcpyAsync(data->getVals(), this->data_vec->mat, getRangeSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));

		// unpin the memory
  CHECK_CUDA_ERROR(cudaHostUnregister(model[0]->getVals()));
  CHECK_CUDA_ERROR(cudaHostUnregister(model[1]->getVals()));
  CHECK_CUDA_ERROR(cudaHostUnregister(data->getVals()));
	
}



