#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <cuda_runtime.h>
#include <CudaKernel.cuh>
#include <cuComplex.h>
#include <prop_kernels.cuh>

using namespace SEP;

class PhaseShift :  public CudaOperator<complex4DReg, complex4DReg> {
public:
    PhaseShift(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range,
                float dz, float eps = 0, bool from_host = true, dim3 grid=1, dim3 block=1);

    void cu_forward (bool add, ComplexVectorMap& model, ComplexVectorMap& data);
    void cu_adjoint (bool add, ComplexVectorMap& model, ComplexVectorMap& data);

    void set_slow(std::complex<float>& sref) {
        _sref_.x = sref.real();
        _sref_.y = sref.imag();
    }

    ~PhaseShift() {
        cudaFree(d_w2);
        cudaFree(d_kx);
        cudaFree(d_ky);
    }

private:
    PS_kernel fwd_kernel;
    PS_kernel adj_kernel;
    cuFloatComplex _sref_;
    float *d_w2, *d_kx, *d_ky;
    float _dz_;
    float _eps_;

    float* fill_in_k(const axis& ax) {
        float *k;
        cudaMalloc((void **)&k, sizeof(float)*ax.n);
        auto h_k = std::vector<float>(ax.n);
	    float dk = 2*M_PI/(ax.d*(ax.n-1));
        for (int ik=0; ik<h_k.size()/2; ik++) 
            h_k[ik] = ik*dk;
        for (int ik=1; ik<h_k.size()/2; ik++) 
            h_k[ax.n-ik] = -h_k[ik];
        cudaMemcpy(k, h_k.data(), sizeof(float)*ax.n, cudaMemcpyHostToDevice);
        return k;
    }

    float* fill_in_w(const axis& ax) {
        float *w;
        cudaMalloc((void **)&w, sizeof(float)*ax.n);
        auto h_w = std::vector<float>(ax.n);
        for (int i=0; i < h_w.size(); ++i) {
            float f = ax.o + i*ax.d;
            f = 2*M_PI*f;
            h_w[i] = f*f;
        }
        cudaMemcpy(w, h_w.data(), sizeof(float)*ax.n, cudaMemcpyHostToDevice);
        return w;
    }

};