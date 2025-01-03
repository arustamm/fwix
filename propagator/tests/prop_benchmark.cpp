#include <complex4DReg.h>
#include "PhaseShift.h"
#include <gtest/gtest.h>
#include <complex_vector.h>
#include <cuda_runtime.h>
#include  <RefSampler.h>
#include <Selector.h>
#include <OneStep.h>
#include <Injection.h>
#include <OneWay.h>

#include <jsonParamObj.h>
#include <random>

class PS_Test : public testing::Test {
 protected:
  void SetUp() override {
    n1 = 1000;
    n2 = 1000;
    n3 = 10;
    n4 = 10;

    // create a vector of slowness values for each frequency
    std::vector<std::complex<float>> slow(n3, {1.f, 0.f});
    auto hyper = std::make_shared<hypercube>(n1, n2, n3, n4);
    space4d = std::make_shared<complex4DReg>(hyper);
    space4d->set(1.f);
    ps = std::make_unique<PhaseShift>(hyper, .1f, 0.f);
    ps->set_slow(slow.data());

    block = {
      {8,8,4}, {8,8,2},
      {16,16,4}, {16,16,2},
      {32,16,2}, {32,4,4}
    };

    grid = {
      {4,4,4}, {4,4,2},
      {8,4,4}, {8,4,2},
      {16,8,4}, {16,8,2},
      {32,16,2}, {32,4,4},
    };

  }

  std::unique_ptr<PhaseShift> ps;
  std::shared_ptr<complex4DReg> space4d;
  std::vector<std::vector<int>> block;
  std::vector<std::vector<int>> grid;
  int n1, n2, n3, n4;
};

TEST_F(PS_Test, fwd) { 
  auto out = space4d->clone();
  
  for (auto& g : grid) {
    for (auto& b : block) {
      ps->set_grid_block({g[0],g[1],g[2]}, {b[0],b[1],b[2]});
      ps->forward(false, space4d, out);
    }
  }
}

TEST_F(PS_Test, fwd_full) { 
  auto out = space4d->clone();
  ps->set_grid_block({8,8,4}, {32,16,2});
  ps->forward(false, space4d, out);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}