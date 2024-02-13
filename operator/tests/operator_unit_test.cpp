#include <complex4DReg.h>
#include "FFT.h"
#include <gtest/gtest.h>
#include <complex_vector.h>
#include <cuda_runtime.h>

class FFTTest : public testing::Test {
 protected:
  void SetUp() override {
    n1 = 100;
    n2 = 100;
    n3 = 20;
    n4 = 10;
    auto hyper = std::make_shared<hypercube>(n1, n2, n3, n4);
    space4d = std::make_shared<complex4DReg>(hyper);
    space4d->set(1.f);
    cuFFT = std::make_unique<cuFFT2d>(hyper, hyper);
  }

  std::unique_ptr<cuFFT2d> cuFFT;
  std::shared_ptr<complex4DReg> space4d;
  int n1, n2, n3, n4;
};

TEST_F(FFTTest, check_flat_index) { 
  int ind1[] = {0,0,0,1};
  int ind2[] = {0,0,1,1};
  int ind3[] = {0,1,0,0};
  int dims[] = {n4, n3, n2, n1};
  ASSERT_EQ(ND_TO_FLAT(ind1, dims), 1);
  ASSERT_EQ(ND_TO_FLAT(ind2, dims), 101);
  ASSERT_EQ(ND_TO_FLAT(ind3, dims), 10000);
}

TEST_F(FFTTest, check_complex_vector) { 
  int ndim = space4d->getHyper()->getNdim();
  int *n = new int[ndim];
  float *d = new float[ndim];
  float *o = new float[ndim];
  cudaMemcpy(n, cuFFT->model_vec["host"]->n, sizeof(int)*ndim, cudaMemcpyDeviceToHost);
  cudaMemcpy(d, cuFFT->model_vec["host"]->d, sizeof(float)*ndim, cudaMemcpyDeviceToHost);
  cudaMemcpy(o, cuFFT->model_vec["host"]->o, sizeof(float)*ndim, cudaMemcpyDeviceToHost);
  for (int i=0; i < ndim; ++i) {
    ASSERT_EQ(n[i], space4d->getHyper()->getAxis(i+1).n);
    ASSERT_EQ(d[i], space4d->getHyper()->getAxis(i+1).d);
    ASSERT_EQ(o[i], space4d->getHyper()->getAxis(i+1).o);
  }
  delete[] n;
  delete[] o;
  delete[] d;
}



TEST_F(FFTTest, forward_inverse) {
  auto input = space4d->clone();
  auto output = space4d->clone();
  auto inv = space4d->clone();
  input->random();
  input->scale(2.f);
  cuFFT->forward(false, input, output);
  cuFFT->adjoint(false, inv, output);
  for (int i = 0; i < space4d->getHyper()->getN123(); ++i) {
    EXPECT_NEAR(input->getVals()[i].real(), inv->getVals()[i].real(), 1e-6);
    EXPECT_NEAR(input->getVals()[i].imag(), inv->getVals()[i].imag(), 1e-6);
  }
}

TEST_F(FFTTest, constant_signal) {
  auto input = space4d->clone();
  auto output = space4d->clone();
  cuFFT->forward(false, input, output);
  for (int i4 = 0; i4 < n4; ++i4) {
    for (int i3 = 0; i3 < n3; ++i3) {
      for (int i2 = 0; i2 < n2; ++i2) {
        for (int i1 = 0; i1 < n1; ++i1) {
          // check DC component
          if (i1 == 0 && i2 == 0) EXPECT_NEAR((*output->_mat)[i4][i3][i2][i1].real(), n1*n2/sqrtf(n1*n2), 1e-6);
          EXPECT_NEAR((*output->_mat)[i4][i3][i2][i1].imag(), 0.0f, 1e-6);
        }
      }
    }
  }
}

TEST_F(FFTTest, mono_plane_wave) {
    int k1 = 10;
    int k2 = 20; 
    auto input = space4d->clone();
    auto output = space4d->clone();
    for (int i4 = 0; i4 < n4; ++i4) {
      for (int i3 = 0; i3 < n3; ++i3) {
        for (int i2 = 0; i2 < n2; ++i2) {
          for (int i1 = 0; i1 < n1; ++i1) {
            float phase = 2 * M_PI * (float(k1*i1)/n1 + float(k2*i2)/n2);
            (*input->_mat)[i4][i3][i2][i1] = std::exp(std::complex<float>(0, phase));
          }
        }
      }
    }
    cuFFT->forward(false, input, output);

    // Check frequency components
    for (int i4 = 0; i4 < n4; ++i4) {
      for (int i3 = 0; i3 < n3; ++i3) {
        for (int i2 = 0; i2 < n2; ++i2) {
          for (int i1 = 0; i1 < n1; ++i1) {
            auto val = (*output->_mat)[i4][i3][i2][i1];
            // expect only a spike at k1, k2
            if (i1 == k1 && i2 == k2) {
              auto conj_val = (*output->_mat)[i4][i3][n2-i2][n1-i1];
              EXPECT_NEAR(val.real(), n1*n2/sqrtf(n1*n2), 1e-6);
            }
          }
        }
      }
    }
}

TEST_F(FFTTest, dotTest) { 
  ASSERT_NO_THROW(cuFFT->dotTest());
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}