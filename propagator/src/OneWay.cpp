
#include <OneWay.h>

using namespace SEP;

void OneWay::one_step_fwd(int iz, complex_vector* __restrict__ wfld) {
	// size_t offset = iz * this->get_wfld_slice_size();
	CHECK_CUDA_ERROR(cudaMemcpyAsync(this->wfld->getVals(), wfld->mat, this->get_wfld_slice_size_in_bytes(), cudaMemcpyDeviceToHost, _stream_));

	std::future<void> fut = this->compress_slice(iz);
	
	prop->set_depth(iz);
	prop->cu_forward(wfld);

	// Wait for compression to finish
	fut.wait();
}

std::future<void> OneWay::compress_slice(int iz) {
	return std::async(std::launch::async, [this, iz]() {

		CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_));

		zfp_field_set_pointer(_zfp_field_, reinterpret_cast<float*>(wfld->getVals()));

		// Estimate maximum size for compressed data
    size_t max_compressed_bufsize = zfp_stream_maximum_size(_zfp_stream_, _zfp_field_);
		// This is temporary because we'll copy it to a smaller buffer later 
    std::unique_ptr<char[]> temp_compressed_data(new char[max_compressed_bufsize]);

		// Open a bitstream and associate it with the compressed data buffer
		bitstream* stream = stream_open(temp_compressed_data.get(), max_compressed_bufsize);
		zfp_stream_set_bit_stream(_zfp_stream_, stream);
		zfp_stream_rewind(_zfp_stream_);
		// Perform compression
		size_t actual_compressed_size = zfp_compress(_zfp_stream_, _zfp_field_);
		stream_close(stream); // Close the bitstream (does not free compressedData buffer)
		if (actual_compressed_size == 0)
			std::cerr << "WARNING: ZFP compression failed for slice " << iz << std::endl;

		// 1. Create a new buffer of the actual compressed size
		// Note: We use a unique_ptr to manage the temporary buffer automatically

		// 2. Allocate a buffer of the actual compressed size
    char* final_compressed_data = new char[actual_compressed_size];
    // 3. Copy data from the temporary buffer to the final buffer
    std::memcpy(final_compressed_data, temp_compressed_data.get(), actual_compressed_size);

		// Check if a buffer already exists and free it to prevent a memory leak
    if (_compressed_wflds_[iz].first != nullptr) 
        delete[] _compressed_wflds_[iz].first;

    // 4. Store the final buffer and its size
    _compressed_wflds_[iz] = {final_compressed_data, actual_compressed_size};
	});
}

void OneWay::decompress_slice(int iz) {
    // Retrieve the compressed data and its actual size for the requested slice
    char* compressed_data_ptr = _compressed_wflds_[iz].first;
    size_t compressed_size = _compressed_wflds_[iz].second;

    if (compressed_data_ptr == nullptr || compressed_size == 0) 
        std::runtime_error("Error: Compressed data for slice " + std::to_string(iz) + " is not available.");

		zfp_field_set_pointer(_zfp_field_, reinterpret_cast<float*>(wfld->getVals()));
    // Open a bitstream using the compressed data buffer
    // The bitstream is read-only for decompression
    bitstream* stream = stream_open(compressed_data_ptr, compressed_size);
    zfp_stream_set_bit_stream(_zfp_stream_, stream);
    zfp_stream_rewind(_zfp_stream_); // Rewind stream to the beginning

    // Perform decompression
    size_t decompressed_size_bytes = zfp_decompress(_zfp_stream_, _zfp_field_);
    stream_close(stream); // Close the bitstream

    if (decompressed_size_bytes == 0) 
			std::cerr << "WARNING: ZFP decompression failed for slice " << iz << std::endl;
}

void OneWay::one_step_adj(int iz, complex_vector* __restrict__ wfld) {
	// propagate one step by changing the state of the wavefield
	prop->set_depth(iz);
	prop->cu_adjoint(wfld);
}

void Downward::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();

	// for (batches in z)

	for (int iz=0; iz < m_ax[3].n-1; ++iz)
		this->one_step_fwd(iz, model);

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
		this->one_step_fwd(iz, model);
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