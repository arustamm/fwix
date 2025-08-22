#pragma once

#include <CudaOperator.h>
#include <complex4DReg.h>
#include <complex3DReg.h>
#include <complex2DReg.h>
#include <Propagator.h>
#include <Scattering.h>  // Assuming this contains DownScattering and UpScattering
#include <memory>
#include <vector>

namespace SEP {

class ExtendedBorn : public CudaOperator<complex4DReg, complex2DReg> {
public:
    ExtendedBorn(
        const std::shared_ptr<hypercube>& domain,
        const std::shared_ptr<hypercube>& range,
        const std::vector<std::shared_ptr<complex4DReg>>& slow_den,
        std::shared_ptr<Propagator> propagator,
        dim3 grid = 1,
        dim3 block = 1,
        cudaStream_t stream = 0
    );

    virtual ~ExtendedBorn() {
      dmodel->~complex_vector();
      CHECK_CUDA_ERROR(cudaFree(dmodel));
      CHECK_CUDA_ERROR(cudaFree(dslow));
      CHECK_CUDA_ERROR(cudaHostUnregister(_slow->getVals()));
      CHECK_CUDA_ERROR(cudaHostUnregister(_den->getVals()));
    };

    // Forward operator: extended Born modeling
    void forward(
        bool add,
        std::vector<std::shared_ptr<complex4DReg>> model,
        std::shared_ptr<complex2DReg> data
    );

    void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data){
      throw std::runtime_error("Not implemented");
    };
	  void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data){
      throw std::runtime_error("Not implemented");
    };

    // You might want to add an adjoint method as well
    // void adjoint(
    //     bool add,
    //     std::vector<std::shared_ptr<complex4DReg>> model,
    //     std::shared_ptr<complex2DReg> data
    // );

private:
    // Get the size of a single depth slice
    size_t getSliceSize() const;

    // Get the size of a single depth slice in bytes
    size_t getSliceSizeInBytes() const;

    // Propagation operators
    std::shared_ptr<Downward> down, bg_down;           // Downward propagator
    std::shared_ptr<Upward> up, bg_up;             // Upward propagator
    std::shared_ptr<Reflect> bg_reflect;        // Reflection operator
    std::shared_ptr<Injection> inj_rec;        // Injection/recording operator
    std::shared_ptr<Propagator> _propagator; // Propagator instance
    std::shared_ptr<complex4DReg> _slow;     // Slowness model
    std::shared_ptr<complex4DReg> _den;      // Density model

    // Scattering operators
    std::shared_ptr<DownScattering> down_scattering;    // Downward scattering
    std::shared_ptr<UpScattering> up_scattering;        // Upward scattering
    std::shared_ptr<BackScattering> back_scattering;        // Back scattering

    // Working arrays
    complex_vector* dmodel;            // Current slowness slice on device
    complex_vector* dslow;             // Current slowness slice on device
    complex_vector* dden;
    std::vector<axis> ax;

    
};

} // namespace SEP