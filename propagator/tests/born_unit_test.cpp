#include <complex4DReg.h>
#include "PhaseShift.h"
#include <gtest/gtest.h>
#include <complex_vector.h>
#include <cuda_runtime.h>
#include  <Scatter.h>
#include <ImagingCondition.h>
#include <ForwardScattering.h>
#include <jsonParamObj.h>
#include <random>

bool verbose = false;
double tolerance = 1e-5;

class Scatter_Test : public testing::Test {
 protected:
  void SetUp() override {
    auto ax = axis(100, 0, 0.01);
    auto ay = axis(100, 0, 0.01);
    auto az = axis(10, 0, 0.01);
    auto aw = axis(15, 1., 0.1);
    auto as = axis(5, 0, 1);;

    auto slow4d = std::make_shared<complex4DReg>(ax, ay, aw, az);
    slow4d->set(1.f);

    // create a vector of slowness values for each frequency
    auto domain = std::make_shared<hypercube>(ax, ay, aw, as);
    space4d = std::make_shared<complex4DReg>(domain);
    space4d->set(1.f);

    Json::Value root;
    auto par = std::make_shared<jsonParamObj>(root);
    dim3 grid = {32, 4, 4};
    dim3 block = {16, 16, 4};
    scatter = std::make_unique<Scatter>(domain, slow4d, par, nullptr, nullptr, grid, block);
    scatter->set_depth(5);
  }

  std::unique_ptr<Scatter> scatter;
  std::shared_ptr<complex4DReg> space4d;
};

TEST_F(Scatter_Test, fwd) { 
  auto out = space4d->clone();
  for (int i=0; i < 3; ++i)
    ASSERT_NO_THROW(scatter->forward(false, space4d, out));
}

TEST_F(Scatter_Test, cu_fwd) { 
  auto out = space4d->clone();
  ASSERT_NO_THROW(scatter->cu_forward(false, scatter->model_vec, scatter->data_vec));
  ASSERT_NO_THROW(scatter->cu_forward(scatter->data_vec));
}

TEST_F(Scatter_Test, dotTest) { 
  auto err = scatter->dotTest(verbose);
  ASSERT_TRUE(err.first <= tolerance);
  ASSERT_TRUE(err.second <= tolerance);
}

class IC_Test : public testing::Test {
 protected:
  void SetUp() override {
    
    ax = {
      axis(100, 0, 0.01), // x-axis
      axis(100, 0, 0.01), // y-axis
      axis(15, 1., 0.1),  // w-axis
      axis(5, 0, 1),       // s-axis
      axis(10, 0, 0.01),  // z-axis
    };

    auto range = std::make_shared<hypercube>(ax[0], ax[1], ax[2], ax[3]);
    wfld1 = std::make_shared<complex4DReg>(range);
    wfld2 = std::make_shared<complex4DReg>(range);

    auto slow4d = std::make_shared<complex4DReg>(ax[0], ax[1], ax[2], ax[4]);
    slow4d->set(1.f);
    dslow = std::make_shared<complex3DReg>(ax[0], ax[1], ax[2]);

    Json::Value root;
    root["nref"] = 3;
    auto par = std::make_shared<jsonParamObj>(root);

    auto down = std::make_shared<Downward>(range, slow4d, par);
    // fill in the background wavefield
    wfld1->random();
    down->forward(false, wfld1, wfld2);

    ic = std::make_unique<ImagingCondition>(dslow->getHyper(), range, down);
  }

  std::unique_ptr<ImagingCondition> ic;
  std::vector<axis> ax;
  std::shared_ptr<complex3DReg> dslow;
  std::shared_ptr<complex4DReg> wfld1, wfld2;
};

TEST_F(IC_Test, cu_fwd) { 
  dslow->set(1.f);
  for (int i=0; i < ax[4].n; ++i) {
    ASSERT_NO_THROW(ic->set_depth(i));
    ASSERT_NO_THROW(ic->forward(false, dslow, wfld1));
    ASSERT_TRUE(std::real(wfld1->dot(wfld1)) > 0.0) << "Forward imaging condition failed at depth " << i;
  }
}

TEST_F(IC_Test, dotTest) { 
  ic->set_depth(5); 
  auto err = ic->dotTest(verbose);
  ASSERT_TRUE(err.first <= tolerance);
  ASSERT_TRUE(err.second <= tolerance);
}

class ForwardScattering_Test : public testing::Test {
 protected:
  void SetUp() override {
    
    ax = {
      axis(100, 0, 0.01), // x-axis
      axis(100, 0, 0.01), // y-axis
      axis(15, 1., 0.1),  // w-axis
      axis(5, 0, 1),       // s-axis
      axis(10, 0, 0.01),  // z-axis
    };

    auto range = std::make_shared<hypercube>(ax[0], ax[1], ax[2], ax[3]);
    wfld1 = std::make_shared<complex4DReg>(range);
    wfld2 = std::make_shared<complex4DReg>(range);

    auto slow4d = std::make_shared<complex4DReg>(ax[0], ax[1], ax[2], ax[4]);
    slow4d->set(1.f);
    dslow = std::make_shared<complex3DReg>(ax[0], ax[1], ax[2]);

    Json::Value root;
    root["nref"] = 3;
    auto par = std::make_shared<jsonParamObj>(root);

    auto down = std::make_shared<Downward>(range, slow4d, par);
    // fill in the background wavefield
    wfld1->random();
    down->forward(false, wfld1, wfld2);

    fscat = std::make_unique<ForwardScattering>(dslow->getHyper(), range, down);
  }

  std::unique_ptr<ForwardScattering> fscat;
  std::vector<axis> ax;
  std::shared_ptr<complex3DReg> dslow;
  std::shared_ptr<complex4DReg> wfld1, wfld2;
};

TEST_F(ForwardScattering_Test, cu_fwd) { 
  dslow->set(1.f);
  for (int i=0; i < ax[4].n; ++i) {
    ASSERT_NO_THROW(fscat->set_depth(i));
    ASSERT_NO_THROW(fscat->forward(false, dslow, wfld1));
    ASSERT_TRUE(std::real(wfld1->dot(wfld1)) > 0.0) << "Forward scattering failed at depth " << i;
  }
}

TEST_F(ForwardScattering_Test, dotTest) { 
  fscat->set_depth(5); 
  auto err = fscat->dotTest(verbose);
  ASSERT_TRUE(err.first <= tolerance);
  ASSERT_TRUE(err.second <= tolerance);
}




int main(int argc, char **argv) {
  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--verbose") {
      verbose = true;
    }
    else if (std::string(argv[i]) == "--tolerance" && i + 1 < argc) {
      tolerance = std::stod(argv[i + 1]);
    }
  }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}