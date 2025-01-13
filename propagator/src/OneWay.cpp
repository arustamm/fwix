
#include <OneWay.h>

using namespace SEP;

void Downward::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();

	// for (batches in z)

	for (int iz=0; iz < m_ax[3].n-1; ++iz) {
	
		// there should be an injection step here
		// inj_src->forward(true, wavelet, temp)
		
		int offset = iz * this->getDomainSize();
		CHECK_CUDA_ERROR(cudaMemcpyAsync(wfld->getVals() + offset, model->mat, getDomainSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));
		// propagate one step by changing the state of the wavefield
		prop->set_depth(iz);
		prop->cu_forward(model);
		
	}

	data->add(model);
	
	// prop->set_slow(slow->next_batch());
	// cudaMemCpyAsync(_slow_chunk, slow->next(), H2D, stream);
// }

}

void Downward::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) model->zero();

	for (int iz=m_ax[3].n-1; iz > 0; --iz) {
		// propagate one step
		prop->set_depth(iz-1);
		prop->cu_adjoint(data);
	}

	model->add(data);


}

void Upward::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	// if(!add) data->zero();
	// for (int iz=ax[4].n-1; iz > 0; --iz) {
	// 	// slice through 5d wfld to get 4d wfld
	// 	model->view_at(curr, iz);
	// 	data->view_at(next, iz-1);
	// 	// propagate one step
	// 	prop->set_depth(iz);
	// 	prop->cu_forward(true, curr, next);
	// 	CHECK_CUDA_ERROR( cudaDeviceSynchronize() );
	// }

}

void Upward::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	// if(!add) model->zero();

	// for (int iz=0; iz < ax[4].n-1; ++iz) {
	// 	// slice through 5d wfld to get 4d wfld
	// 	data->view_at(curr, iz);
	// 	model->view_at(next, iz+1);
	// 	// propagate one step
	// 	prop->set_depth(iz+1);
	// 	prop->cu_adjoint(true, next, curr);
	// 	CHECK_CUDA_ERROR( cudaDeviceSynchronize() );
	// }

}
