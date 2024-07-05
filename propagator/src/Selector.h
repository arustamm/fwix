#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <prop_kernels.cuh>

using namespace SEP;

class Selector : public CudaOperator<complex4DReg,complex4DReg>
{

public:

	Selector(const std::shared_ptr<hypercube>& domain, 
	std::shared_ptr<ComplexVectorMap> model = nullptr, std::shared_ptr<ComplexVectorMap> data = nullptr, dim3 grid=1, dim3 block=1) 
	: CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block) {
		_grid_ = {128, 128, 8};
  	_block_ = {16, 16, 2};

		_size_ = domain->getAxis(1).n * domain->getAxis(2).n * domain->getAxis(3).n;
		CHECK_CUDA_ERROR(cudaMalloc((void **)&d_labels, sizeof(int)*_size_));
		kernel = Selector_kernel(&select_forward, _grid_, _block_);
	};
	
	~Selector() {
		CHECK_CUDA_ERROR(cudaFree(d_labels));
	};

	void set_labels(int* labels) {
		// labels are 3D -- (x,y,w)
		CHECK_CUDA_ERROR(cudaMemcpy(d_labels, labels, sizeof(int)*_size_, cudaMemcpyHostToDevice));
	};
	void set_value(int value) {_value_ = value;}

	void cu_forward(bool add, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data) {
		if (!add) (*data)["host"]->zero();
		kernel.launch((*model)["device"], (*data)["device"], _value_, d_labels);
	};
	void cu_adjoint(bool add, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data) {
		if (!add) (*model)["host"]->zero();
		kernel.launch((*data)["device"], (*model)["device"], _value_, d_labels);
	};

private:
	int _value_;
	int _size_;
	int *d_labels;
	Selector_kernel kernel;

};

