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
    slow4d->random();

    // create a vector of slowness values for each frequency
    auto domain = std::make_shared<hypercube>(nx, ny, nw, ns);
    space4d = std::make_shared<complex4DReg>(domain);
    space4d->set(1.f);

    Json::Value root;
    root["nref"] = nref;
    auto par = std::make_shared<jsonParamObj>(root);
    pspi = std::make_unique<PSPI>(domain, slow4d, par);
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

class Injection_Test : public testing::Test {
 protected:
  void SetUp() override {
    nx = 100;
    auto ax1 = axis(nx, 0.f, 0.01f);
    ny = 100;
    auto ax2 = axis(ny, 0.f, 0.01f);
    nw = 10;
    auto ax3 = axis(nw, 1.f, 1.f);
    ns = 5;
    auto ax4 = axis(ns, 0.f, 1.f);
    nz = 10;
    auto ax5 = axis(nz, 0.f, 0.01f);

    auto range = std::make_shared<hypercube>(ax1, ax2, ax3, ax4, ax5);
    wfld = std::make_shared<complex5DReg>(range);

    int ntrace = 20;
    traces = std::make_shared<complex2DReg>(nw, ntrace);
    auto domain = traces->getHyper();

    std::vector<float> cx(ntrace);
    std::vector<float> cy(ntrace);
    std::vector<float> cz(ntrace);
    std::vector<int> ids(ntrace);

    // Create a random number generator
    std::random_device rd;  // Obtain a random seed from the OS
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> distrib_x(ax1.o + ax1.d, (ax1.n-2)*ax1.d);
    std::uniform_real_distribution<> distrib_y(ax2.o + ax2.d, (ax2.n-2)*ax2.d);
    std::uniform_real_distribution<> distrib_z(ax5.o + ax5.d, (ax5.n-2)*ax5.d);
    std::uniform_real_distribution<> distrib_id(0, ns-1);

    // Generate the random coordinates
    for (int i = 0; i < ntrace; ++i) {
      cx[i] = distrib_x(gen);
      cy[i] = distrib_y(gen);
      cz[i] = distrib_z(gen);
      ids[i] = distrib_id(gen);
    }
    
    injection = std::make_unique<Injection>(domain, range, cx, cy, cz, ids);
  }

  std::unique_ptr<Injection> injection;
  int nx, ny, nz, nw, ns;
  std::shared_ptr<complex5DReg> wfld;
  std::shared_ptr<complex2DReg> traces;
};

TEST_F(Injection_Test, fwd) { 
  for (int i=0; i < 3; ++i)
    ASSERT_NO_THROW(injection->forward(false, traces, wfld));
}

TEST_F(Injection_Test, dotTest) { 
  injection->dotTest();
}

class Downward_Test : public testing::Test {
 protected:
  void SetUp() override {
    nx = 100;
    auto ax1 = axis(nx, 0.f, 0.01f);
    ny = 100;
    auto ax2 = axis(ny, 0.f, 0.01f);
    nw = 10;
    auto ax3 = axis(nw, 1.f, 1.f);
    ns = 5;
    auto ax4 = axis(ns, 0.f, 1.f);
    nz = 10;
    auto ax5 = axis(nz, 0.f, 0.01f);

    auto domain = std::make_shared<hypercube>(ax1, ax2, ax3, ax4, ax5);
    wfld1 = std::make_shared<complex5DReg>(domain);
    wfld2 = std::make_shared<complex5DReg>(domain);

    auto slow4d = std::make_shared<complex4DReg>(nx, ny, nw, nz);
    slow4d->random();

    Json::Value root;
    root["nref"] = 3;
    auto par = std::make_shared<jsonParamObj>(root);

    down = std::make_unique<Downward>(domain, slow4d, par);
  }

  std::unique_ptr<Downward> down;
  int nx, ny, nz, nw, ns;
  std::shared_ptr<complex5DReg> wfld1, wfld2;
};

TEST_F(Downward_Test, fwd) { 
  for (int i=0; i < 3; ++i)
    ASSERT_NO_THROW(down->forward(false, wfld1, wfld2));
}

TEST_F(Downward_Test, adj) { 
  for (int i=0; i < 3; ++i)
    ASSERT_NO_THROW(down->adjoint(false, wfld1, wfld2));
}

TEST_F(Downward_Test, dotTest) { 
  down->dotTest();
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}