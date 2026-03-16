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
#include <cstdint>
#include <cstring>

#if !defined(_WIN32)
  #include <sys/mman.h>
  #include <unistd.h>
#endif

using Vec     = Eigen::VectorXd;
using MatC    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatR    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapMatC = Eigen::Map<const MatC>;
using MapVec  = Eigen::Map<const Vec>;

enum class VDDummyLaw : uint8_t { Spherical = 0, Gaussian = 1 };

struct VDOptions {
  int    T_stop       = 100;
  int    max_vd_proj = 100;
  double eps         = 1e-12;
  bool   standardize = false;
  bool   debug       = false;
  unsigned long long seed = 0ULL;
  VDDummyLaw dummy_law = VDDummyLaw::Spherical;
  double rho = 1.0;
  int mmap_fd         = -1;
  int mmap_block_cols = 0;
};

static constexpr int VD_Y_SENTINEL     = -1;
static constexpr int VD_DUMMY_SENTINEL = -2;

struct ActiveFeature {
  enum class Kind : uint8_t { Real, Dummy };
  Kind kind;
  int  index;
};

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

#if !defined(_WIN32)
inline bool pread_full(int fd, void* buf, std::size_t len, off_t offset) {
  char* p = static_cast<char*>(buf);
  std::size_t remaining = len;
  while (remaining > 0) {
    ssize_t r = ::pread(fd, p, remaining, offset);
    if (r <= 0) return false;
    p += r; offset += r; remaining -= static_cast<std::size_t>(r);
  }
  return true;
}
#endif

inline void advise_sequential(const void* ptr, std::size_t len) {
#if !defined(_WIN32)
  if (ptr && len > 0)
    ::madvise(const_cast<void*>(ptr), len, MADV_SEQUENTIAL);
#else
  (void)ptr; (void)len;
#endif
}

inline void gemv_Xt(const MapMatC& X, const Vec& v, Vec& out,
                    int mmap_fd, int block_cols, double* scratch)
{
  const int n = (int)X.rows(), p = (int)X.cols();
  out.setZero(p);
#if !defined(_WIN32)
  const bool use_pread = (mmap_fd >= 0 && scratch && block_cols > 0);
#else
  const bool use_pread = false;
  (void)mmap_fd; (void)scratch;
#endif
  const int B = (block_cols > 0) ? block_cols : colblock();
  for (int j0 = 0; j0 < p; j0 += B) {
    const int jb = std::min(B, p - j0);
    const double* Xblk;
#if !defined(_WIN32)
    if (use_pread) {
      pread_full(mmap_fd, scratch, std::size_t(jb)*n*sizeof(double),
                 off_t(j0)*n*sizeof(double));
      Xblk = scratch;
    } else
#endif
    { Xblk = X.col(j0).data(); }
    for (int j = 0; j < jb; ++j) {
      const double* x = Xblk + std::ptrdiff_t(j)*n;
      double s = 0.0;
      for (int i = 0; i < n; ++i) s += x[i]*v[i];
      out[j0+j] = s;
    }
  }
}

inline void gemv_Xv(const MapMatC& X, const Vec& v, Vec& y,
                    int mmap_fd, int block_cols, double* scratch)
{
  const int n = (int)X.rows(), p = (int)X.cols();
  y.setZero(n);
#if !defined(_WIN32)
  const bool use_pread = (mmap_fd >= 0 && scratch && block_cols > 0);
#else
  const bool use_pread = false;
  (void)mmap_fd; (void)scratch;
#endif
  const int B = (block_cols > 0) ? block_cols : colblock();
  for (int j0 = 0; j0 < p; j0 += B) {
    const int jb = std::min(B, p - j0);
    const double* Xblk;
#if !defined(_WIN32)
    if (use_pread) {
      pread_full(mmap_fd, scratch, std::size_t(jb)*n*sizeof(double),
                 off_t(j0)*n*sizeof(double));
      Xblk = scratch;
    } else
#endif
    { Xblk = X.col(j0).data(); }
    for (int j = 0; j < jb; ++j) {
      const double* x = Xblk + std::ptrdiff_t(j)*n;
      const double w = v[j0+j];
      if (w == 0.0) continue;
      for (int i = 0; i < n; ++i) y[i] += x[i]*w;
    }
  }
}

inline void gemv_Xt(const MapMatC& X, const Vec& v, Vec& out) {
  gemv_Xt(X, v, out, -1, 0, nullptr);
}
inline void gemv_Xv(const MapMatC& X, const Vec& v, Vec& y) {
  gemv_Xv(X, v, y, -1, 0, nullptr);
}

inline double gamma_mt(double m, std::mt19937_64& rng) {
    auto uniform01 = [&](){
      constexpr double scale = 1.0/(1ULL<<53);
      return (rng()>>(64-53))*scale;
    };
    std::normal_distribution<double> normal(0.0,1.0);
    if (m < 1.0) {
      double g = gamma_mt(m+1.0, rng);
      double u = std::max(uniform01(), std::numeric_limits<double>::min());
      return g*std::pow(u, 1.0/m);
    }
    const double d=m-1.0/3.0, c=1.0/std::sqrt(9.0*d);
    for (;;) {
      double z=normal(rng), v=1.0+c*z;
      if (v<=0.0) continue;
      double v3=v*v*v;
      if (uniform01()<1.0-0.0331*(z*z)*(z*z)) return d*v3;
      double u=std::max(uniform01(), std::numeric_limits<double>::min());
      if (std::log(u)<0.5*z*z+d*(1.0-v3+std::log(v3))) return d*v3;
    }
}

} // namespace vd_detail
