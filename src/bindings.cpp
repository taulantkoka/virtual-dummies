// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vd_lars.hpp"
#include "vd_omp.hpp"
#include "vd_afs.hpp"

namespace py = pybind11;

PYBIND11_MODULE(vd_selectors, m) {

  // ==================================================================
  // Shared types
  // ==================================================================

  py::enum_<VDDummyLaw>(m, "VDDummyLaw", py::arithmetic())
    .value("Spherical", VDDummyLaw::Spherical)
    .value("Gaussian",  VDDummyLaw::Gaussian)
    .export_values();

  py::class_<VDOptions>(m, "VDOptions")
    .def(py::init<>())
    .def_readwrite("T_max",       &VDOptions::T_max)
    .def_readwrite("max_vd_proj", &VDOptions::max_vd_proj)
    .def_readwrite("eps",         &VDOptions::eps)
    .def_readwrite("standardize", &VDOptions::standardize)
    .def_readwrite("debug",       &VDOptions::debug)
    .def_readwrite("seed",        &VDOptions::seed)
    .def_readwrite("dummy_law",   &VDOptions::dummy_law)
    .def_readwrite("rho",         &VDOptions::rho);

  // ==================================================================
  // VD_LARS
  // ==================================================================

  py::class_<VD_LARS::ActiveFeature>(m, "LARSActiveFeature")
    .def_property_readonly("kind", [](const VD_LARS::ActiveFeature& af) {
        return static_cast<int>(af.kind);  // 0 = Real, 1 = Dummy
    })
    .def_readonly("index", &VD_LARS::ActiveFeature::index)
    .def("__repr__", [](const VD_LARS::ActiveFeature& af) {
        return std::string("<LARSActiveFeature kind=") +
               (af.kind == VD_LARS::ActiveFeature::Kind::Real ? "Real" : "Dummy") +
               " index=" + std::to_string(af.index) + ">";
    });

  py::class_<VD_LARS>(m, "VD_LARS")
    .def(py::init([](py::array_t<double, py::array::f_style> X,
                     py::array_t<double, py::array::c_style | py::array::forcecast> y,
                     int num_dummies,
                     const VDOptions& opt) {
          auto bx = X.request();
          auto by = y.request();
          if (bx.ndim != 2) throw std::runtime_error("X must be 2D (Fortran/col-major).");
          if (by.ndim != 1) throw std::runtime_error("y must be 1D.");
          if (bx.shape[0] != by.shape[0]) throw std::runtime_error("X.rows != y.size");
          return std::unique_ptr<VD_LARS>(new VD_LARS(
              static_cast<double*>(bx.ptr), (int)bx.shape[0], (int)bx.shape[1],
              static_cast<double*>(by.ptr), (int)by.shape[0],
              num_dummies, opt));
        }),
        py::arg("X"), py::arg("y"), py::arg("num_dummies"), py::arg("options"),
        py::keep_alive<1,2>(), py::keep_alive<1,3>())

    .def("run", [](VD_LARS& self, int T) {
          py::gil_scoped_release release;
          return self.run(T);
        }, py::arg("T") = 1)

    .def("basis_size",           &VD_LARS::basis_size)
    .def("num_dummies",          &VD_LARS::num_dummies)
    .def("num_realized_dummies", &VD_LARS::num_realized_dummies)
    .def("n_features",           &VD_LARS::n_features)
    .def("n_samples",            &VD_LARS::n_samples)

    .def("normx",    &VD_LARS::normx_view,   py::return_value_policy::reference_internal)
    .def("vd_proj",  &VD_LARS::vd_proj_view, py::return_value_policy::reference_internal)
    .def("vd_corr",  &VD_LARS::vd_corr_view, py::return_value_policy::reference_internal)
    .def("vd_stick", &VD_LARS::vd_stick_view,py::return_value_policy::reference_internal)

    .def("beta_view_copy",         &VD_LARS::beta_view_copy)
    .def("beta_real",              &VD_LARS::beta_real)
    .def("corr_view_copy",         &VD_LARS::corr_view_copy)
    .def("corr_realized_view",     &VD_LARS::corr_realized_view_copy)
    .def("active_indices",         &VD_LARS::active_indices)
    .def("is_dummy_realized",      &VD_LARS::is_dummy_realized_view)

    .def("active_features", [](const VD_LARS& self) {
        py::list out;
        for (const auto& af : self.active_features_copy()) {
          const char* kind =
              (af.kind == VD_LARS::ActiveFeature::Kind::Real) ? "real" : "dummy";
          out.append(py::make_tuple(kind, af.index));
        }
        return out;
    });

  // ==================================================================
  // VD_OMP
  // ==================================================================

  py::class_<VD_OMP::ActiveFeature>(m, "OMPActiveFeature")
    .def_property_readonly("kind", [](const VD_OMP::ActiveFeature& af) {
        return static_cast<int>(af.kind);
    })
    .def_readonly("index", &VD_OMP::ActiveFeature::index)
    .def("__repr__", [](const VD_OMP::ActiveFeature& af) {
        return std::string("<OMPActiveFeature kind=") +
               (af.kind == VD_OMP::ActiveFeature::Kind::Real ? "Real" : "Dummy") +
               " index=" + std::to_string(af.index) + ">";
    });

  py::class_<VD_OMP>(m, "VD_OMP")
    .def(py::init([](py::array_t<double, py::array::f_style> X,
                     py::array_t<double, py::array::c_style | py::array::forcecast> y,
                     int num_dummies,
                     const VDOptions& opt) {
          auto bx = X.request();
          auto by = y.request();
          if (bx.ndim != 2) throw std::runtime_error("X must be 2D (Fortran/col-major).");
          if (by.ndim != 1) throw std::runtime_error("y must be 1D.");
          if (bx.shape[0] != by.shape[0]) throw std::runtime_error("X.rows != y.size");
          return std::unique_ptr<VD_OMP>(new VD_OMP(
              static_cast<double*>(bx.ptr), (int)bx.shape[0], (int)bx.shape[1],
              static_cast<double*>(by.ptr), (int)by.shape[0],
              num_dummies, opt));
        }),
        py::arg("X"), py::arg("y"), py::arg("num_dummies"), py::arg("options"),
        py::keep_alive<1,2>(), py::keep_alive<1,3>())

    .def("run", [](VD_OMP& self, int T) {
          py::gil_scoped_release release;
          return self.run(T);
        }, py::arg("T") = 1)

    .def("basis_size",           &VD_OMP::basis_size)
    .def("num_dummies",          &VD_OMP::num_dummies)
    .def("num_realized_dummies", &VD_OMP::num_realized_dummies)
    .def("n_features",           &VD_OMP::n_features)
    .def("n_samples",            &VD_OMP::n_samples)

    .def("normx",    &VD_OMP::normx_view,   py::return_value_policy::reference_internal)
    .def("vd_proj",  &VD_OMP::vd_proj_view, py::return_value_policy::reference_internal)
    .def("vd_corr",  &VD_OMP::vd_corr_view, py::return_value_policy::reference_internal)
    .def("vd_stick", &VD_OMP::vd_stick_view,py::return_value_policy::reference_internal)

    .def("beta_view_copy",         &VD_OMP::beta_view_copy)
    .def("beta_real",              &VD_OMP::beta_real)
    .def("corr_view_copy",         &VD_OMP::corr_view_copy)
    .def("corr_realized_view",     &VD_OMP::corr_realized_view_copy)
    .def("active_indices",         &VD_OMP::active_indices)
    .def("is_dummy_realized",      &VD_OMP::is_dummy_realized_view)

    .def("active_features", [](const VD_OMP& self) {
        py::list out;
        for (const auto& af : self.active_features_copy()) {
          const char* kind =
              (af.kind == VD_OMP::ActiveFeature::Kind::Real) ? "real" : "dummy";
          out.append(py::make_tuple(kind, af.index));
        }
        return out;
    });

  // ==================================================================
  // VD_AFS
  // ==================================================================

  py::class_<VD_AFS::ActiveFeature>(m, "AFSActiveFeature")
    .def_property_readonly("kind", [](const VD_AFS::ActiveFeature& af) {
        return static_cast<int>(af.kind);
    })
    .def_readonly("index", &VD_AFS::ActiveFeature::index)
    .def("__repr__", [](const VD_AFS::ActiveFeature& af) {
        return std::string("<AFSActiveFeature kind=") +
               (af.kind == VD_AFS::ActiveFeature::Kind::Real ? "Real" : "Dummy") +
               " index=" + std::to_string(af.index) + ">";
    });

  py::class_<VD_AFS>(m, "VD_AFS")
    .def(py::init([](py::array_t<double, py::array::f_style> X,
                     py::array_t<double, py::array::c_style | py::array::forcecast> y,
                     int num_dummies,
                     const VDOptions& opt) {
          auto bx = X.request();
          auto by = y.request();
          if (bx.ndim != 2) throw std::runtime_error("X must be 2D (Fortran/col-major).");
          if (by.ndim != 1) throw std::runtime_error("y must be 1D.");
          if (bx.shape[0] != by.shape[0]) throw std::runtime_error("X.rows != y.size");
          return std::unique_ptr<VD_AFS>(new VD_AFS(
              static_cast<double*>(bx.ptr), (int)bx.shape[0], (int)bx.shape[1],
              static_cast<double*>(by.ptr), (int)by.shape[0],
              num_dummies, opt));
        }),
        py::arg("X"), py::arg("y"), py::arg("num_dummies"), py::arg("options"),
        py::keep_alive<1,2>(), py::keep_alive<1,3>())

    .def("run", [](VD_AFS& self, int T) {
          py::gil_scoped_release release;
          return self.run(T);
        }, py::arg("T") = 1)

    .def("basis_size",           &VD_AFS::basis_size)
    .def("num_dummies",          &VD_AFS::num_dummies)
    .def("num_realized_dummies", &VD_AFS::num_realized_dummies)
    .def("n_features",           &VD_AFS::n_features)
    .def("n_samples",            &VD_AFS::n_samples)
    .def("rho",                  &VD_AFS::rho)

    .def("normx",    &VD_AFS::normx_view,   py::return_value_policy::reference_internal)
    .def("vd_proj",  &VD_AFS::vd_proj_view, py::return_value_policy::reference_internal)
    .def("vd_corr",  &VD_AFS::vd_corr_view, py::return_value_policy::reference_internal)
    .def("vd_stick", &VD_AFS::vd_stick_view,py::return_value_policy::reference_internal)

    .def("beta_view_copy",         &VD_AFS::beta_view_copy)
    .def("beta_real",              &VD_AFS::beta_real)
    .def("corr_view_copy",         &VD_AFS::corr_view_copy)
    .def("corr_realized_view",     &VD_AFS::corr_realized_view_copy)
    .def("active_indices",         &VD_AFS::active_indices)
    .def("is_dummy_realized",      &VD_AFS::is_dummy_realized_view)

    .def("active_features", [](const VD_AFS& self) {
        py::list out;
        for (const auto& af : self.active_features_copy()) {
          const char* kind =
              (af.kind == VD_AFS::ActiveFeature::Kind::Real) ? "real" : "dummy";
          out.append(py::make_tuple(kind, af.index));
        }
        return out;
    });
}