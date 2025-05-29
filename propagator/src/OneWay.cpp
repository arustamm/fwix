
#include <OneWay.h>

using namespace SEP;

void OneWay::one_step_fwd(int iz, complex_vector* __restrict__ wfld) {
	size_t offset = iz * this->get_wfld_slice_size();
	CHECK_CUDA_ERROR(cudaMemcpyAsync(this->wfld->getVals() + offset, wfld->mat, this->get_wfld_slice_size_in_bytes(), cudaMemcpyDeviceToHost, _stream_));
	// CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_));
	// propagate one step by changing the state of the wavefield
	prop->set_depth(iz);
	prop->cu_forward(wfld);
}

// void OneWay::stream_to_disk() {
    
// 	size_t outSize;
//     char *compressedData = SZ_compress(conf, wfld->getVals(), outSize);
    
//     // Save both compressed arrays
//     std::ofstream outfile(_filename_, std::ios::binary);
    
//     // Write size information
//     size_t real_size = compressed_real.size();
//     size_t imag_size = compressed_imag.size();
//     outfile.write(reinterpret_cast<char*>(&real_size), sizeof(real_size));
//     outfile.write(reinterpret_cast<char*>(&imag_size), sizeof(imag_size));
    
//     // Write compressed data
//     outfile.write(reinterpret_cast<char*>(compressed_real.data()), real_size);
//     outfile.write(reinterpret_cast<char*>(compressed_imag.data()), imag_size);
//     outfile.close();
// }

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