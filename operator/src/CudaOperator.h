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
								std::shared_ptr<ComplexVectorMap> model = nullptr, std::shared_ptr<ComplexVectorMap> data = nullptr, dim3 grid=1, dim3 block=1) 
	: _grid_(grid), _block_(block) {
		
		setDomainRange(domain, range);
		if (model == nullptr) {
			model_alloc = true;
			model_vec = make_complex_vector_map(domain, _grid_, _block_);
		}
		else model_vec = model;

		if (data == nullptr) {
			data_alloc = true;
			data_vec = make_complex_vector_map(range, _grid_, _block_);
		}
		else data_vec = data;
	 };

	virtual ~CudaOperator() {
		if (model_alloc) {
			CHECK_CUDA_ERROR(cudaFree((*model_vec)["device"]));
			delete (*model_vec)["host"];
		}
		if (data_alloc) {
			CHECK_CUDA_ERROR(cudaFree((*data_vec)["device"]));
			delete (*data_vec)["host"];
		}
	};

	void set_grid(dim3 grid) {_grid_ = grid;};
	void set_block(dim3 block) {_block_ = block;};

	virtual void cu_forward(bool add, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data) = 0;
	virtual void cu_adjoint(bool add, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data) = 0;


	void forward(bool add, std::shared_ptr<M>& model, std::shared_ptr<D>& data) {
		if (!add) data->zero();
		CHECK_CUDA_ERROR(cudaMemcpyAsync((*model_vec)["host"]->mat, model->getVals(), sizeof(cuFloatComplex) * getDomainSize(), cudaMemcpyHostToDevice));
		if (add) CHECK_CUDA_ERROR(cudaMemcpyAsync((*data_vec)["host"]->mat, data->getVals(), sizeof(cuFloatComplex) * getDomainSize(), cudaMemcpyHostToDevice));
		cu_forward(add, model_vec, data_vec);
		CHECK_CUDA_ERROR(cudaMemcpyAsync(data->getVals(), (*data_vec)["host"]->mat, sizeof(cuFloatComplex) * getRangeSize(), cudaMemcpyDeviceToHost));
	};

	// this is host-to-host function
	void adjoint(bool add, std::shared_ptr<M>& model, std::shared_ptr<D>& data) {
		if (!add) model->zero();
		CHECK_CUDA_ERROR(cudaMemcpyAsync((*data_vec)["host"]->mat,data->getVals(),sizeof(cuFloatComplex) * getRangeSize(), cudaMemcpyHostToDevice));
		if (add) CHECK_CUDA_ERROR(cudaMemcpyAsync((*model_vec)["host"]->mat, model->getVals(), sizeof(cuFloatComplex) * getDomainSize(), cudaMemcpyHostToDevice));
		cu_adjoint(add, model_vec, data_vec);
		CHECK_CUDA_ERROR(cudaMemcpyAsync(model->getVals(),(*model_vec)["host"]->mat,sizeof(cuFloatComplex) * getDomainSize(), cudaMemcpyDeviceToHost));
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

std::shared_ptr<ComplexVectorMap> model_vec, data_vec; 

protected:
	std::shared_ptr<hypercube> _domain;
	std::shared_ptr<hypercube> _range;
	dim3 _grid_, _block_;
	bool model_alloc = false, data_alloc = false;
	
};

