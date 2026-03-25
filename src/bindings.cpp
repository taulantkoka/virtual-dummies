// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vd_lars.hpp"
#include "vd_omp.hpp"
#include "vd_afs.hpp"
#include "vd_afs_logistic.hpp"
#include "trex.hpp"
#include "memory_mapped_eigen_matrix.hpp"

namespace py = pybind11;

// Helper: construct any VD solver from numpy arrays
template <typename Solver>
static std::unique_ptr<Solver> make_solver(
    py::array_t<double, py::array::f_style> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    int num_dummies, const VDOptions& opt)
{
  auto bx = X.request();
  auto by = y.request();
  if (bx.ndim != 2) throw std::runtime_error("X must be 2D.");
  if (by.ndim != 1) throw std::runtime_error("y must be 1D.");
  if (bx.shape[0] != by.shape[0]) throw std::runtime_error("X.rows != y.size");
  return std::unique_ptr<Solver>(new Solver(
      static_cast<double*>(bx.ptr), (int)bx.shape[0], (int)bx.shape[1],
      static_cast<double*>(by.ptr), (int)by.shape[0],
      num_dummies, opt));
}

// Helper: bind common accessors on any VD solver (they all inherit from VD_Base)
template <typename Cls, typename PyClass>
static void bind_common(PyClass& c) {
  c.def("basis_size",           [](const Cls& s) { return s.basis_size(); })
   .def("num_dummies",          [](const Cls& s) { return s.num_dummies(); })
   .def("num_realized_dummies", [](const Cls& s) { return s.num_realized_dummies(); })
   .def("n_features",           [](const Cls& s) { return s.n_features(); })
   .def("n_samples",            [](const Cls& s) { return s.n_samples(); })
 
   .def("normx",    [](Cls& s) -> const Vec&  { return s.normx_view(); },
        py::return_value_policy::reference_internal)
   .def("vd_proj",  [](Cls& s) -> const MatR& { return s.vd_proj_view(); },
        py::return_value_policy::reference_internal)
   .def("vd_corr",  [](Cls& s) -> const Vec&  { return s.vd_corr_view(); },
        py::return_value_policy::reference_internal)
   .def("vd_stick", [](Cls& s) -> const Vec&  { return s.vd_stick_view(); },
        py::return_value_policy::reference_internal)
 
   .def("beta_view_copy",     [](const Cls& s) { return s.beta_view_copy(); })
   .def("beta_real",          [](const Cls& s) { return s.beta_real(); })
   .def("corr_view_copy",     [](const Cls& s) { return s.corr_view_copy(); })
   .def("corr_realized_view", [](const Cls& s) { return s.corr_realized_view_copy(); })
   .def("active_indices",     [](const Cls& s) { return s.active_indices(); })
   .def("is_dummy_realized",  [](const Cls& s) { return s.is_dummy_realized_view(); })
 
   .def("active_features", [](const Cls& self) {
       py::list out;
       for (const auto& af : self.active_features_copy()) {
         const char* kind =
             (af.kind == ActiveFeature::Kind::Real) ? "real" : "dummy";
         out.append(py::make_tuple(kind, af.index));
       }
       return out;
   });
}
 

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
    .def_readwrite("T_stop",           &VDOptions::T_stop)
    .def_readwrite("max_vd_proj",     &VDOptions::max_vd_proj)
    .def_readwrite("eps",             &VDOptions::eps)
    .def_readwrite("standardize",     &VDOptions::standardize)
    .def_readwrite("debug",           &VDOptions::debug)
    .def_readwrite("seed",            &VDOptions::seed)
    .def_readwrite("dummy_law",       &VDOptions::dummy_law)
    .def_readwrite("rho",             &VDOptions::rho)
    .def_readwrite("mmap_fd",         &VDOptions::mmap_fd)
    .def_readwrite("mmap_block_cols", &VDOptions::mmap_block_cols);

  py::class_<ActiveFeature>(m, "ActiveFeature")
    .def_property_readonly("kind", [](const ActiveFeature& af) {
        return static_cast<int>(af.kind);
    })
    .def_readonly("index", &ActiveFeature::index)
    .def("__repr__", [](const ActiveFeature& af) {
        return std::string("<ActiveFeature kind=") +
               (af.kind == ActiveFeature::Kind::Real ? "Real" : "Dummy") +
               " index=" + std::to_string(af.index) + ">";
    });

  // ==================================================================
  // VD_LARS
  // ==================================================================
  auto lars = py::class_<VD_LARS>(m, "VD_LARS")
    .def(py::init(&make_solver<VD_LARS>),
        py::arg("X"), py::arg("y"), py::arg("num_dummies"), py::arg("options"),
        py::keep_alive<1,2>(), py::keep_alive<1,3>())
    .def("run", [](VD_LARS& self, int T) {
          py::gil_scoped_release release;
          return self.run(T);
        }, py::arg("T") = 1);
  bind_common<VD_LARS>(lars);

  // ==================================================================
  // VD_OMP
  // ==================================================================
  auto omp = py::class_<VD_OMP>(m, "VD_OMP")
    .def(py::init(&make_solver<VD_OMP>),
        py::arg("X"), py::arg("y"), py::arg("num_dummies"), py::arg("options"),
        py::keep_alive<1,2>(), py::keep_alive<1,3>())
    .def("run", [](VD_OMP& self, int T) {
          py::gil_scoped_release release;
          return self.run(T);
        }, py::arg("T") = 1);
  bind_common<VD_OMP>(omp);

  // ==================================================================
  // VD_AFS
  // ==================================================================
  auto afs = py::class_<VD_AFS>(m, "VD_AFS")
    .def(py::init(&make_solver<VD_AFS>),
        py::arg("X"), py::arg("y"), py::arg("num_dummies"), py::arg("options"),
        py::keep_alive<1,2>(), py::keep_alive<1,3>())
    .def("run", [](VD_AFS& self, int T) {
          py::gil_scoped_release release;
          return self.run(T);
        }, py::arg("T") = 1)
    .def("rho", &VD_AFS::rho);
  bind_common<VD_AFS>(afs);

  // ==================================================================
  // VD_AFS_Logistic
  // ==================================================================
  auto afs_log = py::class_<VD_AFS_Logistic>(m, "VD_AFS_Logistic")
    .def(py::init(&make_solver<VD_AFS_Logistic>),
        py::arg("X"), py::arg("y"), py::arg("num_dummies"), py::arg("options"),
        py::keep_alive<1,2>(), py::keep_alive<1,3>())
    .def("run", [](VD_AFS_Logistic& self, int T) {
          py::gil_scoped_release release;
          return self.run(T);
        }, py::arg("T") = 1);
  bind_common<VD_AFS_Logistic>(afs_log);

  // ==================================================================
  // MMapMatrix
  // ==================================================================
  using MMap = MemoryMappedEigenMatrix<double>;

  py::class_<MMap>(m, "MMapMatrix")
    .def(py::init([](const std::string& filename, int nrows, int ncols, bool writable) {
          auto mode = writable ? MMap::Mode::ReadWrite : MMap::Mode::ReadOnly;
          return std::unique_ptr<MMap>(new MMap(filename, nrows, ncols, mode));
        }),
        py::arg("filename"), py::arg("nrows"), py::arg("ncols"),
        py::arg("writable") = false)

    .def_property_readonly("nrows", &MMap::nrows)
    .def_property_readonly("ncols", &MMap::ncols)
    .def_property_readonly("nbytes", &MMap::size_bytes)
    .def_property_readonly("shape", [](const MMap& self) {
        return py::make_tuple(self.nrows(), self.ncols());
    })

    .def("as_array", [](py::object self_obj) {
      auto& self = self_obj.cast<MMap&>();
      py::ssize_t n = self.nrows(), p = self.ncols();
      py::ssize_t itemsize = sizeof(double);
      return py::array(py::dtype::of<double>(),
          {n, p}, {itemsize, n*itemsize}, self.data(), self_obj);
    }, "Zero-copy NumPy view onto the mapped matrix.")

    .def("fileno", &MMap::fileno,
        "File descriptor (Unix) or -1 (Windows).")

    .def_static("create_from_array",
        [](const std::string& filename,
           py::array_t<double, py::array::f_style | py::array::forcecast> arr) {
          auto buf = arr.request();
          if (buf.ndim != 2) throw std::runtime_error("Array must be 2D.");
          MMap::create_from_ptr(filename,
              static_cast<const double*>(buf.ptr),
              buf.shape[0], buf.shape[1], true, MMap::Mode::ReadOnly);
        },
        py::arg("filename"), py::arg("array"),
        "Write a col-major numpy array to a binary file.")

    .def("__repr__", [](const MMap& self) {
        return "<MMapMatrix " + std::to_string(self.nrows()) + "x" +
               std::to_string(self.ncols()) + " (" +
               std::to_string(self.size_bytes()/(1024*1024)) + " MB) " +
               (self.mode() == MMap::Mode::ReadOnly ? "ro" : "rw") + ">";
    });

  // ==================================================================
  // TRex
  // ==================================================================
  py::enum_<SolverType>(m, "SolverType")
    .value("LARS", SolverType::LARS)
    .value("OMP",  SolverType::OMP)
    .value("AFS",  SolverType::AFS)
    .value("AFS_Logistic", SolverType::AFS_Logistic)
    .export_values();

  py::enum_<CalibMode>(m, "CalibMode")
    .value("FixedTL",       CalibMode::FixedTL)
    .value("CalibrateT",    CalibMode::CalibrateT)
    .value("CalibrateL",    CalibMode::CalibrateL)
    .value("CalibrateBoth", CalibMode::CalibrateBoth)
    .export_values();

  py::class_<TRexOptions>(m, "TRexOptions")
    .def(py::init<>())
    .def_readwrite("tFDR",           &TRexOptions::tFDR)
    .def_readwrite("K",              &TRexOptions::K)
    .def_readwrite("L_factor",       &TRexOptions::L_factor)
    .def_readwrite("T_stop",          &TRexOptions::T_stop)
    .def_readwrite("max_L_factor",   &TRexOptions::max_L_factor)
    .def_readwrite("stride_width",   &TRexOptions::stride_width)
    .def_readwrite("posthoc_mode",   &TRexOptions::posthoc_mode)
    .def_readwrite("max_stale_strides", &TRexOptions::max_stale_strides)
    .def_readwrite("max_vd_proj",    &TRexOptions::max_vd_proj)
    .def_readwrite("eps",            &TRexOptions::eps)
    .def_readwrite("verbose",        &TRexOptions::verbose)
    .def_readwrite("seed",           &TRexOptions::seed)
    .def_readwrite("solver",         &TRexOptions::solver)
    .def_readwrite("calib",          &TRexOptions::calib)
    .def_readwrite("dummy_law",      &TRexOptions::dummy_law)
    .def_readwrite("rho",            &TRexOptions::rho)
    .def_readwrite("mmap_fd",        &TRexOptions::mmap_fd)
    .def_readwrite("mmap_block_cols",&TRexOptions::mmap_block_cols)
    .def_readwrite("n_threads",      &TRexOptions::n_threads);

  py::class_<TRexResult>(m, "TRexResult")
    .def(py::init<>())
    .def_readonly("selected_var",  &TRexResult::selected_var)
    .def_readonly("v_thresh",      &TRexResult::v_thresh)
    .def_readonly("T_stop",        &TRexResult::T_stop)
    .def_readonly("num_dummies",   &TRexResult::num_dummies)
    .def_readonly("L_calibrated",  &TRexResult::L_calibrated)
    .def_readonly("V",             &TRexResult::V)
    .def_readonly("FDP_hat_mat",   &TRexResult::FDP_hat_mat)
    .def_readonly("Phi_mat",       &TRexResult::Phi_mat)
    .def_readonly("Phi_prime",     &TRexResult::Phi_prime)
    .def_readonly("K",             &TRexResult::K)
    .def("to_dict", [](const TRexResult& r) {
        py::dict d;
        d["selected_var"]  = r.selected_var;
        d["v_thresh"]      = r.v_thresh;
        d["T_stop"]        = r.T_stop;
        d["num_dummies"]   = r.num_dummies;
        d["L_calibrated"]  = r.L_calibrated;
        d["V"]             = r.V;
        d["FDP_hat_mat"]   = r.FDP_hat_mat;
        d["Phi_mat"]       = r.Phi_mat;
        d["Phi_prime"]     = r.Phi_prime;
        d["K"]             = r.K;
        return d;
    });

  py::class_<TRexSelector>(m, "TRexSelector")
    .def(py::init<const TRexOptions&>(), py::arg("options"))
    .def("run",
        [](TRexSelector& self,
           py::array_t<double, py::array::f_style | py::array::forcecast> X,
           py::array_t<double, py::array::c_style | py::array::forcecast> y) {
          auto bx = X.request();
          auto by = y.request();
          if (bx.ndim != 2) throw std::runtime_error("X must be 2D.");
          if (by.ndim != 1) throw std::runtime_error("y must be 1D.");
          if (bx.shape[0] != by.shape[0]) throw std::runtime_error("X.rows != y.size");
          Eigen::Map<const MatC> Xm(static_cast<double*>(bx.ptr),
              (int)bx.shape[0], (int)bx.shape[1]);
          Eigen::Map<const Vec> ym(static_cast<double*>(by.ptr), (int)by.shape[0]);
          py::gil_scoped_release release;
          return self.run(Xm, ym);
        },
        py::arg("X"), py::arg("y"),
        py::keep_alive<1,2>(), py::keep_alive<1,3>());
}