#pragma once
#include <cufft.h>
#include <cufftXt.h>
#include "CudaOperator.h"
#include "fft_callback.cuh"
#include <complex4DReg.h>

using namespace SEP;

class cuFFT2d : public CudaOperator<complex4DReg, complex4DReg> {
	public:
		cuFFT2d(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range, bool from_host = true);
		
		~cuFFT2d() {
			cufftDestroy(plan);
		};

		// this is on-device functions
		void cu_forward(bool add, ComplexVectorMap& model, ComplexVectorMap& data);
		void cu_adjoint(bool add, ComplexVectorMap& model, ComplexVectorMap& data);

	private:
		cufftHandle plan;
		int NX, NY, BATCH, SIZE; 

		void register_ortho_callback() { 
			auto h_storeCallbackPtr = get_host_callback_ptr();
			cufftXtSetCallback(plan, (void **)&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX, (void **)&(model_vec["host"]->n));
		}
		

};
