#pragma once

#include <CudaOperator.h>
#include <complex4DReg.h>
#include <OneWay.h>
#include <ImagingCondition.h>
#include <Scatter.h>
#include <memory>

namespace SEP {

class ForwardScattering : public CudaOperator<complex3DReg, complex4DReg> {
public:
    ForwardScattering(
        const std::shared_ptr<hypercube>& domain,
				const std::shared_ptr<hypercube>& range,
        std::shared_ptr<OneWay> oneway,
        complex_vector* model = nullptr, 
        complex_vector* data = nullptr, 
        dim3 grid = 1, 
        dim3 block = 1, 
        cudaStream_t stream = 0
    );

    virtual ~ForwardScattering() {
        temp_vec->~complex_vector();
        CHECK_CUDA_ERROR(cudaFree(temp_vec));
    };

    // Set the depth for all operators (imaging condition, scatter, propagator)
    void set_depth(int iz);

    // Forward scattering: model -> scattered wavefield -> propagated data
    void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) override;

    // Adjoint scattering: data -> back-propagated -> adjoint scatter -> adjoint imaging -> model
    void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) override;

private:
    std::shared_ptr<OneStep> prop;              // Propagator from OneWay
    std::shared_ptr<Scatter> sc;                // Scattering operator
    std::shared_ptr<ImagingCondition> ic;       // Imaging condition operator
    complex_vector* temp_vec;                // Temporary vector for intermediate results
};

} // namespace SEP