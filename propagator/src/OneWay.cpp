
#include <OneWay.h>

using namespace SEP;

void OneWay::one_step_fwd(int iz, complex_vector* __restrict__ wfld) {
	int offset = iz * this->getDomainSize();
	CHECK_CUDA_ERROR(cudaMemcpyAsync(this->wfld->getVals() + offset, wfld->mat, getDomainSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));
	// propagate one step by changing the state of the wavefield
	prop->set_depth(iz);
	prop->cu_forward(wfld);
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