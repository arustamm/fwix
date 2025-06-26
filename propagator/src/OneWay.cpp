
#include <OneWay.h>

using namespace SEP;

void OneWay::one_step_fwd(int iz, complex_vector* __restrict__ wfld) {
	// Select a buffer and event from the pool in a round-robin fashion
  int pool_idx = iz % _wfld_pool.size();
	auto event = _events_pool[pool_idx];
	auto wfld_buffer = _wfld_pool[pool_idx];
	
  // 1. Copy data to the pooled host buffer
  CHECK_CUDA_ERROR(cudaMemcpyAsync(wfld_buffer->getVals(), wfld->mat, this->get_wfld_slice_size_in_bytes(), cudaMemcpyDeviceToHost, _stream_));

  // 2. Record an event in the stream *after* the copy
  CHECK_CUDA_ERROR(cudaEventRecord(event, _stream_));

  // 3. Launch compression, passing it the buffer and event it needs
  auto fut = this->compress_slice(iz, pool_idx, event);
  _compression_futures.push(std::move(fut)); // Store the future

  // 4. Launch the next GPU propagation step immediately
  prop->set_depth(iz);
  prop->cu_forward(wfld);
  
  // 5. NO fut.wait()! Return immediately and let the loop continue.
}

std::future<void> OneWay::compress_slice(int iz, int pool_idx, cudaEvent_t event) {
  return std::async(std::launch::async, [this, iz, pool_idx, event]() {
    // Get the dedicated resources for this compression task
    auto wfld_buffer = _wfld_pool[pool_idx];
    auto zfp_s = _zfp_stream_pool[pool_idx];
    auto zfp_f = _zfp_field_pool[pool_idx];

    // Wait for the D->H copy to finish for this specific slice
    CHECK_CUDA_ERROR(cudaEventSynchronize(event));

    // Associate the zfp field with the data buffer for this task
    zfp_field_set_pointer(zfp_f, reinterpret_cast<float*>(wfld_buffer->getVals()));

    // Estimate maximum size for compressed data
    size_t max_compressed_bufsize = zfp_stream_maximum_size(zfp_s, zfp_f);

    // Create the final compressed buffer directly as a vector.
    // This handles all memory allocation safely.
    std::vector<char> final_compressed_data(max_compressed_bufsize);

    // Open a bitstream and associate it with the vector's underlying buffer
    bitstream* stream = stream_open(final_compressed_data.data(), final_compressed_data.size());
    zfp_stream_set_bit_stream(zfp_s, stream);
    zfp_stream_rewind(zfp_s);

    // Perform compression directly into the vector's memory
    size_t actual_compressed_size = zfp_compress(zfp_s, zfp_f);
    stream_close(stream);

    if (actual_compressed_size == 0) 
      throw std::runtime_error("Compress: Failed for wavefield for slice " + std::to_string(iz));

    // Shrink the vector to the actual size used, freeing unused memory.
    final_compressed_data.resize(actual_compressed_size);

    // Store the vector of compressed data.
    // A mutex is needed here to protect access to the shared _compressed_wflds_ vector,
    // making the operation thread-safe.
		_compressed_wflds_[iz] = std::move(final_compressed_data);
  });
}

void OneWay::decompress_slice(int iz) {
    const auto& compressed_data = _compressed_wflds_[iz];
    
    if (compressed_data.empty()) {
        throw std::runtime_error("Error: Compressed data for slice " + std::to_string(iz) + " is not available.");
    }

    // Use a dedicated pool index for decompression or make it thread-safe
    static std::mutex decompress_mutex;
    std::lock_guard<std::mutex> lock(decompress_mutex);
    
    auto wfld_buffer = _wfld_pool[0];
    auto zfp_s = _zfp_stream_pool[0];
    auto zfp_f = _zfp_field_pool[0];

    // Reset the ZFP stream for decompression
    zfp_stream_close(zfp_s);
    zfp_s = zfp_stream_open(NULL);
    double rel_error_bound = 1E-6; // Use the same error bound as compression
    zfp_stream_set_accuracy(zfp_s, rel_error_bound);
    _zfp_stream_pool[0] = zfp_s;

    zfp_field_set_pointer(zfp_f, reinterpret_cast<float*>(wfld_buffer->getVals()));

    bitstream* stream = stream_open((void*)compressed_data.data(), compressed_data.size());
    zfp_stream_set_bit_stream(zfp_s, stream);
    zfp_stream_rewind(zfp_s);

    if (!zfp_decompress(zfp_s, zfp_f)) {
        stream_close(stream);
        throw std::runtime_error("ZFP decompression failed for slice " + std::to_string(iz));
    }

    stream_close(stream);
}

void OneWay::one_step_adj(int iz, complex_vector* __restrict__ wfld) {
	// propagate one step by changing the state of the wavefield
	prop->set_depth(iz);
	prop->cu_adjoint(wfld);
}

void OneWay::check_pipeline() {
	// If the pipeline is full...
	if (_compression_futures.size() == _wfld_pool.size()) {
			// ...wait for the OLDEST compression job to finish.
			// This frees up its host buffer for re-use.
			_compression_futures.front().wait();
			_compression_futures.pop();
	}
}

void OneWay::wait_for_pipeline() {
	// After the loop, wait for the last few remaining tasks in the pipeline to complete.
	while (!_compression_futures.empty()) {
			_compression_futures.front().wait();
			_compression_futures.pop();
	}
}

void Downward::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();

	for (int iz=0; iz < m_ax[3].n-1; ++iz) {
		this->check_pipeline();
		// Enqueue more work!
		this->one_step_fwd(iz, model);
	}

	this->wait_for_pipeline();

	data->add(model);
	
	// prop->set_slow(slow->next_batch());
	// cudaMemCpyAsync(_slow_chunk, slow->next(), H2D, stream);
// }

}

void Downward::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) model->zero();

	for (int iz=m_ax[3].n-1; iz > 0; --iz) {
		// propagate one step
		this->one_step_adj(iz-1, data);
	}

	model->add(data);


}

void Upward::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();

	// for (batches in z)

	for (int iz=m_ax[3].n-1; iz > 0; --iz) {
		// If the pipeline is full...
		if (_compression_futures.size() == _wfld_pool.size()) {
				// ...wait for the OLDEST compression job to finish.
				// This frees up its host buffer for re-use.
				_compression_futures.front().wait();
				_compression_futures.pop();
		}
		// Enqueue more work!
		this->one_step_fwd(iz, model);
	}

	// After the loop, wait for the last few remaining tasks in the pipeline to complete.
	while (!_compression_futures.empty()) {
			_compression_futures.front().wait();
			_compression_futures.pop();
	}

	data->add(model);

}

void Upward::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) model->zero();

	for (int iz=0; iz < m_ax[3].n-1; ++iz) {
		// propagate one step
		this->one_step_adj(iz+1, data);
	}

	model->add(data);

}


// void Upward::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

// 	if(!add) data->zero();

// 	// for (batches in z)

// 	for (int iz=m_ax[3].n-1; iz > 0; --iz) {

// 		// TODO: fix that, the reflected and propagated wavefeilds should not be the same
	
// 		int offset = iz * this->getDomainSize();
// 		// save the upward going wavefield for future gradient calculation
// 		CHECK_CUDA_ERROR(cudaMemcpyAsync(wfld->getVals() + offset, model->mat, getDomainSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));

// 		CHECK_CUDA_ERROR(cudaMemcpyAsync(reflected->mat, _down_->get_wfld()->getVals() + offset, getDomainSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
// 		// reflect the downward wavefield	
// 		reflect->set_depth(iz);
// 		reflect->cu_forward(reflected);
// 		// propagate one step by changing the state of the wavefield
// 		prop->set_depth(iz);
// 		prop->cu_forward(add, reflected, data);
		
// 	}

// 	data->add(model);

// }