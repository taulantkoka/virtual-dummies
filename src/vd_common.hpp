// vd_common.hpp
// Shared types, options, and helpers for the VD-* family of solvers.
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <optional>
#include <utility>
#include <unordered_set>
#include <limits>
#include <tuple>
#include <cmath>
#include <cstdlib>

// ---------- Type aliases ----------
using Vec     = Eigen::VectorXd;
using MatC    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatR    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapMatC = Eigen::Map<const MatC>;
using MapVec  = Eigen::Map<const Vec>;

// ---------- Dummy law ----------
enum class VDDummyLaw : uint8_t { Spherical = 0, Gaussian = 1 };

// ---------- Options ----------
struct VDOptions {
  int    T_max       = 100;     // max realized dummies stored
  int    max_vd_proj = 100;     // max rows in vd_proj (basis projections)
  double eps         = 1e-12;
  bool   standardize = false;
  bool   debug       = false;
  unsigned long long seed = 0ULL;

  VDDummyLaw dummy_law = VDDummyLaw::Spherical;

  double rho = 1.0;   // AFS shrinkage: 1.0 = FS/OMP, (0,1) = AFS blend
};

// ---------- Sentinel values for basis_indices_ ----------
static constexpr int VD_Y_SENTINEL     = -1;
static constexpr int VD_DUMMY_SENTINEL = -2;

// ---------- Streaming GEMV helpers (bounded working set) ----------
namespace vd_detail {

inline int colblock() {
  static int B = [](){
    const char* s = std::getenv("VD_COLBLOCK");
    long v = s ? std::strtol(s, nullptr, 10) : 0;
    if (v <= 0) v = 8192;
    return int(v);
  }();
  return B;
}

// out = X^T * v   (out size p)
inline void gemv_Xt(const MapMatC& X, const Vec& v, Vec& out) {
  const int n = (int)X.rows();
  const int p = (int)X.cols();
  out.setZero(p);
  const int B = colblock();
  for (int j0 = 0; j0 < p; j0 += B) {
    const int jb = std::min(B, p - j0);
    const double* Xblk = X.col(j0).data();
    for (int j = 0; j < jb; ++j) {
      const double* x = Xblk + std::ptrdiff_t(j) * n;
      double s = 0.0;
      for (int i = 0; i < n; ++i) s += x[i] * v[i];
      out[j0 + j] = s;
    }
  }
}

// y = X * v   (y size n)
inline void gemv_Xv(const MapMatC& X, const Vec& v, Vec& y) {
  const int n = (int)X.rows();
  const int p = (int)X.cols();
  y.setZero(n);
  const int B = colblock();
  for (int j0 = 0; j0 < p; j0 += B) {
    const int jb = std::min(B, p - j0);
    const double* Xblk = X.col(j0).data();
    for (int j = 0; j < jb; ++j) {
      const double* x = Xblk + std::ptrdiff_t(j) * n;
      const double w = v[j0 + j];
      if (w == 0.0) continue;
      for (int i = 0; i < n; ++i) y[i] += x[i] * w;
    }
  }
}

// Marsaglia-Tsang Gamma(m,1)
inline double gamma_mt(double m, std::mt19937_64& rng) {
    auto uniform01 = [&](){
        constexpr int bits = 53;
        constexpr double scale = 1.0 / (1ULL << bits);
        uint64_t x = rng() >> (64 - bits);
        return x * scale;
    };
    std::normal_distribution<double> normal(0.0, 1.0);

    if (m < 1.0) {
        const double g = gamma_mt(m + 1.0, rng);
        const double u = std::max(uniform01(), std::numeric_limits<double>::min());
        return g * std::pow(u, 1.0 / m);
    }

    const double d = m - 1.0/3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);

    for (;;) {
        const double z = normal(rng);
        const double v = 1.0 + c * z;
        if (v <= 0.0) continue;
        const double v3 = v*v*v;
        if (uniform01() < 1.0 - 0.0331 * (z*z)*(z*z))
            return d * v3;
        const double u = std::max(uniform01(), std::numeric_limits<double>::min());
        if (std::log(u) < 0.5*z*z + d*(1.0 - v3 + std::log(v3)))
            return d * v3;
    }
}

} // namespace vd_detail