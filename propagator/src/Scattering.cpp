#include <Scattering.h>

using namespace SEP;

ForwardScattering::ForwardScattering(
	const std::shared_ptr<hypercube>& domain,
	const std::shared_ptr<hypercube>& range,
	const std::shared_ptr<complex4DReg>& slow,
	std::shared_ptr<OneWay> oneway,
	complex_vector* model, complex_vector* data, 
  dim3 grid, dim3 block, cudaStream_t stream
) : CudaOperator<complex3DReg, complex4DReg>(domain, range, model, data, grid, block, stream) {
	
	_grid_ = {32, 4, 4};
	_block_ = {16, 16, 4};
	
	temp_vec = data_vec->cloneSpace();
	temp_vec->set_grid_block(_grid_, _block_);

	prop = oneway->getPropagator();
	ic = std::make_shared<ImagingCondition>(domain, range, oneway, model_vec, temp_vec, _grid_, _block_, _stream_);
	sc = std::make_shared<Scatter>(range, slow, oneway->getPar(), temp_vec, temp_vec, _grid_, _block_, _stream_);
}

void ForwardScattering::set_depth(int iz) {
	// Set the depth for the imaging condition
	ic->set_depth(iz);
	sc->set_depth(iz);
	prop->set_depth(iz);
}

void UpScattering::set_depth(int iz) {
	// Set the depth for the imaging condition
	ic->set_depth(iz);
	if (iz > 0) {
		sc->set_depth(iz-1);
		prop->set_depth(iz-1);
	}
}

void ForwardScattering::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();

	// IC contains information about the background wavefield
	ic->cu_forward(false, model, temp_vec);
	sc->cu_forward(temp_vec);
 	prop->cu_forward(true, temp_vec, data);

}

void ForwardScattering::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) model->zero();

  	prop->cu_adjoint(false, temp_vec, data);
	sc->cu_adjoint(temp_vec);
	ic->cu_adjoint(true, model, temp_vec);
}

BackScattering::BackScattering(
	const std::shared_ptr<hypercube>& domain,
	const std::shared_ptr<hypercube>& range,
	const std::vector<std::shared_ptr<complex4DReg>>& slow_impedance,
	std::shared_ptr<OneWay> oneway,
	complex_vector* model, complex_vector* data, 
  dim3 grid, dim3 block, cudaStream_t stream
) : CudaOperator<complex5DReg, complex4DReg>(domain, range, model, data, grid, block, stream) {

	_grid_ = {32, 4, 4};
	_block_ = {16, 16, 4};

	auto ax = domain->getAxes();
	auto subhyper = std::make_shared<hypercube>(std::vector<axis>{ax[0], ax[1], ax[2]});
	
	temp_vec = make_complex_vector(subhyper, _grid_, _block_, _stream_);
	temp_vec->set_grid_block(_grid_, _block_);

	lin_refl = std::make_shared<LinReflect>(domain, subhyper, slow_impedance, model_vec, temp_vec, _grid_, _block_, _stream_);
	ic = std::make_shared<ImagingCondition>(subhyper, range, oneway, temp_vec, data_vec, _grid_, _block_, _stream_);
	
}

void BackScattering::set_depth(int iz) {
	// Set the depth for the imaging condition
	ic->set_depth(iz);
	lin_refl->set_depth(iz);
}


void BackScattering::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();

	lin_refl->cu_forward(false, model, temp_vec);
	// IC contains information about the background wavefield
	ic->cu_forward(true, temp_vec, data);
 	
}

void BackScattering::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) model->zero();

  	ic->cu_adjoint(false, temp_vec, data);
	lin_refl->cu_adjoint(true, model, temp_vec);

}

