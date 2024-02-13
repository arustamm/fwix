#include <complex4DReg.h>
#include "PhaseShift.h"
#include <gtest/gtest.h>
#include <complex_vector.h>
#include <cuda_runtime.h>

class PS_Test : public testing::Test {
 protected:
  void SetUp() override {
    n1 = 1000;
    n2 = 1000;
    n3 = 10;
    n4 = 1;
    slow = {1.f, 0.f};
    auto hyper = std::make_shared<hypercube>(n1, n2, n3, n4);
    space4d = std::make_shared<complex4DReg>(hyper);
    space4d->set(1.f);
    ps = std::make_unique<PhaseShift>(hyper, hyper, .1f, 0.f);
    ps->set_slow(slow);
  }

  std::unique_ptr<PhaseShift> ps;
  std::shared_ptr<complex4DReg> space4d;
  int n1, n2, n3, n4;
  std::complex<float> slow;
};

TEST_F(PS_Test, fwd) { 
  for (int i=0; i < 100; ++i)
    ps->forward(true, space4d, space4d);
}

TEST_F(PS_Test, adj) { 
  for (int i=0; i < 100; ++i)
    ps->adjoint(true, space4d, space4d);
}




TEST_F(PS_Test, dotTest) { 
  ps->dotTest();

  ps->set_grid({2,2,2});
  ps->set_block({2,2,2});
  ps->dotTest();

  ps->set_grid({4,4,4});
  ps->set_block({4,4,4});
  ps->dotTest();
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}