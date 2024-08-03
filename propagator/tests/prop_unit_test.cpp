#include <complex4DReg.h>
#include "PhaseShift.h"
#include <gtest/gtest.h>
#include <complex_vector.h>
#include <cuda_runtime.h>
#include  <RefSampler.h>
#include <Selector.h>
#include <OneStep.h>
#include <jsonParamObj.h>

class PS_Test : public testing::Test {
 protected:
  void SetUp() override {
    n1 = 1000;
    n2 = 1000;
    n3 = 10;
    n4 = 1;

    // create a vector of slowness values for each frequency
    std::vector<std::complex<float>> slow(n3, {1.f, 0.f});
    auto hyper = std::make_shared<hypercube>(n1, n2, n3, n4);
    space4d = std::make_shared<complex4DReg>(hyper);
    space4d->set(1.f);
    ps = std::make_unique<PhaseShift>(hyper, .1f, 0.f);
    ps->set_slow(slow.data());
  }

  std::unique_ptr<PhaseShift> ps;
  std::shared_ptr<complex4DReg> space4d;
  int n1, n2, n3, n4;
};

TEST_F(PS_Test, fwd) { 
  auto out = space4d->clone();
  ps->set_grid({1024, 1024, 16});
  ps->set_block({2, 2, 2});
  for (int i=0; i < 100; ++i)
    ps->forward(false, space4d, out);
}

TEST_F(PS_Test, adj) { 
  auto out = space4d->clone();
  for (int i=0; i < 100; ++i)
    ps->adjoint(false, out, space4d);
}

TEST_F(PS_Test, dotTest) { 
  ps->dotTest();
}

class Selector_Test : public testing::Test {
 protected:
  void SetUp() override {
    nx = 100;
    ny = 100;
    nz = 10;
    nw = 10;
    ns = 1;
    nref = 3;

    auto slow4d = std::make_shared<complex4DReg>(nx, ny, nw, nz);
    ref = std::make_shared<RefSampler>(slow4d, nref);
    slow4d->random();

    // create a vector of slowness values for each frequency
    auto domain = std::make_shared<hypercube>(nx, ny, nw, ns);
    space4d = std::make_shared<complex4DReg>(domain);
    space4d->set(1.f);

    select = std::make_unique<Selector>(domain);
  }

  std::unique_ptr<Selector> select;
  std::shared_ptr<complex4DReg> space4d;
  std::shared_ptr<RefSampler> ref;
  int nx, ny, nz, nw, ns, nref;
};

TEST_F(Selector_Test, dotTest) { 
  for (int iz = 0; iz < 3; ++iz) {
    select->set_labels(ref->get_ref_labels(iz));
    for (int iref = 0; iref < nref; ++iref) {
      select->set_value(iref);
      ASSERT_NO_THROW(select->dotTest());
    }
  }
};

class PSPI_Test : public testing::Test {
 protected:
  void SetUp() override {
    nx = 100;
    ny = 100;
    nz = 10;
    nw = 10;
    ns = 1;
    int nref = 3;

    auto slow4d = std::make_shared<complex4DReg>(nx, ny, nw, nz);
    auto ref = std::make_shared<RefSampler>(slow4d, nref);
    slow4d->random();

    // create a vector of slowness values for each frequency
    auto domain = std::make_shared<hypercube>(nx, ny, nw, ns);
    space4d = std::make_shared<complex4DReg>(domain);
    space4d->set(1.f);

    Json::Value root;
    root["nref"] = nref;
    auto par = std::make_shared<jsonParamObj>(root);
    pspi = std::make_unique<PSPI>(domain, slow4d, par, ref);
    pspi->set_depth(5);
  }

  std::unique_ptr<PSPI> pspi;
  std::shared_ptr<complex4DReg> space4d;
  int nx, ny, nz, nw, ns;
};

TEST_F(PSPI_Test, fwd) { 
  auto out = space4d->clone();
  for (int i=0; i < 3; ++i)
    ASSERT_NO_THROW(pspi->forward(false, space4d, out));
}

TEST_F(PSPI_Test, dotTest) { 
  pspi->dotTest();
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}