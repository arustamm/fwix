#pragma once
#include <vector>
#include <floatHyper.h>
#include <complexHyper.h>
#include <complex_vector.h>

using namespace SEP;

template <class M, class D>
class CudaOperator
{
public:
	CudaOperator(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range, 
								complex_vector* model = nullptr, complex_vector* data = nullptr, dim3 grid=1, dim3 block=1) 
	: _grid_(grid), _block_(block) {
		
		setDomainRange(domain, range);
		if (model == nullptr) {
			model_alloc = true;
			model_vec = make_complex_vector(domain, _grid_, _block_);
		}
		else model_vec = model;

		if (data == nullptr) {
			data_alloc = true;
			data_vec = make_complex_vector(range, _grid_, _block_);
		}
		else data_vec = data;
	 };

	virtual ~CudaOperator() {
		if (model_alloc) {
			CHECK_CUDA_ERROR(cudaFree(model_vec));
		}
		if (data_alloc) {
			CHECK_CUDA_ERROR(cudaFree(data_vec));
		}
	};

	void set_grid(dim3 grid) {_grid_ = grid;};
	void set_block(dim3 block) {_block_ = block;};

	virtual void cu_forward(bool add, const complex_vector* __restrict__ model, complex_vector* __restrict__ data) = 0;
	virtual void cu_adjoint(bool add, complex_vector* __restrict__ model, const complex_vector* __restrict__ data) = 0;


	void forward(bool add, std::shared_ptr<M>& model, std::shared_ptr<D>& data) {
		// pin the host memory
		CHECK_CUDA_ERROR(cudaHostRegister(model->getVals(), getDomainSizeInBytes(), cudaHostRegisterDefault));
		CHECK_CUDA_ERROR(cudaHostRegister(data->getVals(), getRangeSizeInBytes(), cudaHostRegisterDefault));

		if (add) {
			CHECK_CUDA_ERROR(cudaMemcpyAsync(data_vec->mat, data->getVals(), getRangeSizeInBytes(), cudaMemcpyHostToDevice));
		}
		else {
			data->zero();
		}
		
		CHECK_CUDA_ERROR(cudaMemcpyAsync(model_vec->mat, model->getVals(), getDomainSizeInBytes(), cudaMemcpyHostToDevice));
		cu_forward(add, model_vec, data_vec);
		CHECK_CUDA_ERROR(cudaMemcpyAsync(data->getVals(), data_vec->mat, getRangeSizeInBytes(), cudaMemcpyDeviceToHost));

		// unpin the memory
		CHECK_CUDA_ERROR(cudaHostUnregister(model->getVals()));
		CHECK_CUDA_ERROR(cudaHostUnregister(data->getVals()));
	};

	// this is host-to-host function
	void adjoint(bool add, std::shared_ptr<M>& model, std::shared_ptr<D>& data) {
		// pin the host memory
		CHECK_CUDA_ERROR(cudaHostRegister(model->getVals(), getDomainSizeInBytes(), cudaHostRegisterDefault));
		CHECK_CUDA_ERROR(cudaHostRegister(data->getVals(), getRangeSizeInBytes(), cudaHostRegisterDefault));

		if (add) {
			CHECK_CUDA_ERROR(cudaMemcpyAsync(model_vec->mat, model->getVals(), getDomainSizeInBytes(), cudaMemcpyHostToDevice));
		}
		else {
			model->zero();
		}

		CHECK_CUDA_ERROR(cudaMemcpyAsync(data_vec->mat,data->getVals(), getRangeSizeInBytes(), cudaMemcpyHostToDevice));
		cu_adjoint(add, model_vec, data_vec);
		CHECK_CUDA_ERROR(cudaMemcpyAsync(model->getVals(),model_vec->mat, getDomainSizeInBytes(), cudaMemcpyDeviceToHost));

		// unpin the memory
		CHECK_CUDA_ERROR(cudaHostUnregister(model->getVals()));
		CHECK_CUDA_ERROR(cudaHostUnregister(data->getVals()));
	};

	const std::shared_ptr<hypercube>& getDomain() const{
		return _domain;
	}
	const std::shared_ptr<hypercube>& getRange() const{
		return _range;
	}
	const int getDomainSize() const{
		return _domain->getN123();
	}
	const int getRangeSize() const{
		return _range->getN123();
	}
	const size_t getDomainSizeInBytes() const{
		return sizeof(cuFloatComplex)*_domain->getN123();
	}
	const size_t getRangeSizeInBytes() const{
		return sizeof(cuFloatComplex)*_range->getN123();
	}

	void setDomainRange(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range) {
		_domain = domain->clone();
		_range = range->clone();
	}

	void dotTest() {
		std::shared_ptr<M> m1 = std::make_shared<M>(getDomain());
		std::shared_ptr<D> d1 = std::make_shared<D>(getRange());
		auto _model = m1->clone();
		auto _data = d1->clone();
		_model->random();
		_data->random();

		forward(false, _model,d1);
		adjoint(false, m1,_data);

		// std::cerr << typeid(*this).name() << '\n';
    double err;
		std::cout << "********** ADD = FALSE **********" << '\n';
		std::cout << "<m,A'd>: " << _model->dot(m1) << std::endl;
		std::cout << "<Am,d>: " << std::conj(_data->dot(d1)) << std::endl;
    err = std::real(std::conj(_data->dot(d1))/_model->dot(m1)) -1.;
    if (std::abs(err) > 1e-3) throw std::runtime_error("Error exceeds tolerance: " + std::to_string(err));
    else std::cout << "Passed with relative error: " << std::to_string(err) << std::endl;
		std::cout << "*********************************" << '\n';

		forward(true, _model,d1);
		adjoint(true, m1,_data);

		std::cout << "********** ADD = TRUE **********" << '\n';
		std::cout << "<m,A'd>: " << _model->dot(m1) << std::endl;
		std::cout << "<Am,d>: " << std::conj(_data->dot(d1)) << std::endl;
    err = std::real(std::conj(_data->dot(d1))/_model->dot(m1)) -1.;
    if (std::abs(err) > 1e-3) throw std::runtime_error("Error exceeds tolerance: " + std::to_string(err));
    else std::cout << "Passed with relative error: " << std::to_string(err) << std::endl;
		std::cout << "*********************************" << '\n';
};

complex_vector *model_vec, *data_vec; 

protected:
	std::shared_ptr<hypercube> _domain;
	std::shared_ptr<hypercube> _range;
	dim3 _grid_, _block_;
	bool model_alloc = false, data_alloc = false;
	
};

