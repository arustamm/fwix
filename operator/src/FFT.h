#pragma once
#include <cufft.h>
#include <cufftXt.h>
#include "CudaOperator.h"
#include "fft_callback.cuh"
#include <complex4DReg.h>

using namespace SEP;

class cuFFT2d : public CudaOperator<complex4DReg, complex4DReg> {
	public:
		cuFFT2d(const std::shared_ptr<hypercube>& domain, std::shared_ptr<ComplexVectorMap> model = nullptr, std::shared_ptr<ComplexVectorMap> data = nullptr);
		
		~cuFFT2d() {
			cufftDestroy(plan);
		};

		// this is on-device functions
		void cu_forward(bool add, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data);
		void cu_adjoint(bool add, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data);
		void cu_forward(std::shared_ptr<ComplexVectorMap> data);
		void cu_adjoint(std::shared_ptr<ComplexVectorMap> data);

	private:
		cufftHandle plan;
		int NX, NY, BATCH, SIZE; 
		std::shared_ptr<ComplexVectorMap> temp;

		void register_ortho_callback() { 
			auto h_storeCallbackPtr = get_host_callback_ptr();
			cufftXtSetCallback(plan, (void **)&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX, (void **)&((*model_vec)["host"]->n));
		}
		

};
