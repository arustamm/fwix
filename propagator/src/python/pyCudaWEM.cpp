#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "PhaseShift.h"
#include "RefSampler.h"
#include "OneStep.h"
#include "Injection.h"

namespace py = pybind11;

using namespace SEP;

PYBIND11_MODULE(pyCudaWEM, clsOps) {

py::class_<PhaseShift, std::shared_ptr<PhaseShift>>(clsOps, "PhaseShift")
    .def(py::init<std::shared_ptr<hypercube>, float, float &>(),
        "Initialize PhaseShift")

    .def("forward",
        (void (PhaseShift::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        PhaseShift::forward,
        "Forward operator of PhaseShift")

    .def("adjoint",
        (void (PhaseShift::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        PhaseShift::adjoint,
        "Adjoint operator of PhaseShift")

    .def("set_slow", [](PhaseShift &self, py::array_t<std::complex<float>, py::array::c_style> arr) {
            auto buf = arr.request();
            self.set_slow(static_cast<std::complex<float> *>(buf.ptr));
        });

py::class_<RefSampler, std::shared_ptr<RefSampler>>(clsOps, "RefSampler")
    .def(py::init<std::shared_ptr<complex4DReg>&, int>(),
        "Initialize RefSampler")

    .def("get_ref_slow", [](RefSampler &self, int iz, int iref) {
        return py::array_t<std::complex<float>>(
            {self._nw_}, // shape
            self.get_ref_slow(iz, iref) // pointer to data
        );
    })

    .def("get_ref_labels", [](RefSampler &self, int iz) {
        return py::array_t<int>(
            {self._nw_, self._ny_, self._nx_}, // shape
            self.get_ref_labels(iz) // pointer to data
        );
    });

py::class_<PSPI, std::shared_ptr<PSPI>>(clsOps, "PSPI")
    .def(py::init<std::shared_ptr<hypercube>&, std::shared_ptr<complex4DReg>, std::shared_ptr<paramObj>, std::shared_ptr<RefSampler>>(),
        "Initialize PSPI")

    .def("forward",
        (void (PSPI::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        PSPI::forward,
        "Forward operator of PSPI")

    .def("adjoint",
        (void (PSPI::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        PSPI::adjoint,
        "Adjoint operator of PSPI")

    .def("set_depth", 
        (void (PSPI::*)(int)) &
        PSPI::set_depth,
        "Set depth of PSPI");

py::class_<NSPS, std::shared_ptr<NSPS>>(clsOps, "NSPS")
    .def(py::init<std::shared_ptr<hypercube>&, std::shared_ptr<complex4DReg>, std::shared_ptr<paramObj>, std::shared_ptr<RefSampler>>(),
        "Initialize NSPS")

    .def("forward",
        (void (NSPS::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        NSPS::forward,
        "Forward operator of NSPS")

    .def("adjoint",
        (void (NSPS::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        NSPS::adjoint,
        "Adjoint operator of NSPS")

    .def("set_depth", 
        (void (NSPS::*)(int)) &
        NSPS::set_depth,
        "Set depth of NSPS");


py::class_<Injection, std::shared_ptr<Injection>>(clsOps, "Injection")
    .def(py::init<std::shared_ptr<hypercube>&, std::shared_ptr<hypercube>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<int>&>(),
        "Initialize Injection")

    .def("forward",
        (void (Injection::*)(bool, std::shared_ptr<complex2DReg>&, std::shared_ptr<complex5DReg>&)) &
        Injection::forward,
        "Forward operator of Injection")

    .def("adjoint",
        (void (Injection::*)(bool, std::shared_ptr<complex2DReg>&, std::shared_ptr<complex5DReg>&)) &
        Injection::adjoint,
        "Adjoint operator of Injection")

    .def("set_coords", 
        (void (Injection::*)(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<int>&)) &
        Injection::set_coords,
        "Set depth of Injection");


}

