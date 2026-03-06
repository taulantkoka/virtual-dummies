// vd_lars.cpp
#include "vd_lars.hpp"
using DummyLaw = VDDummyLaw;

#if defined(__has_include)
  #if __has_include(<cblas.h>)
    #pragma message("CBLAS detected")
    #include <cblas.h>
    #define USE_CBLAS 1
  #endif
#endif

// ---------- GEMV helpers ----------
// static inline void gemv_Xt_vec(const MapMatC& X_, const Vec& v, Vec& out) {
// #if defined(USE_CBLAS)
//   cblas_dgemv(CblasColMajor, CblasTrans, (int)X_.rows(), (int)X_.cols(),
//               1.0, X_.data(), (int)X_.rows(), v.data(), 1, 0.0, out.data(), 1);
// #else
//   out.noalias() = X_.transpose() * v;
// #endif
// }

// static inline void gemv_X_vec(const MapMatC& X_, const Vec& v, Vec& out) {
// #if defined(USE_CBLAS)
//   cblas_dgemv(CblasColMajor, CblasNoTrans,
//               (int)X_.rows(), (int)X_.cols(),
//               1.0, X_.data(), (int)X_.rows(),
//               v.data(), 1,
//               0.0, out.data(), 1);
// #else
//   out.noalias() = X_ * v;
// #endif
// }

// ---------- Streaming GEMV helpers (bounded working set) ----------
// Tunables: block columns; can be set via env VD_COLBLOCK or fallback to 8192
static inline int vd_colblock() {
  static int B = [](){
    const char* s = std::getenv("VD_COLBLOCK");
    long v = s ? std::strtol(s, nullptr, 10) : 0;
    if (v <= 0) v = 8192; // ~64MB for n=1e3; tune per machine/cache
    return int(v);
  }();
  return B;
}

// X is column-major map of size n×p.
// out = Xᵀ * v   (out size p)
static inline void gemv_Xt_vec(const MapMatC& X_, const Vec& v, Vec& out) {
  const int n = (int)X_.rows();
  const int p = (int)X_.cols();
  out.setZero(p);

  const int B = vd_colblock();
  for (int j0 = 0; j0 < p; j0 += B) {
    const int jb = std::min(B, p - j0);
    // sub-block X[:, j0:j0+jb)
    const double* Xblk = X_.col(j0).data(); // contiguous block (n×jb), col-major

#if defined(USE_CBLAS)
    // Use GEMV per block: (n×jb)ᵀ * (n) => (jb)
    // out[j0:j0+jb] = Xblkᵀ * v
    cblas_dgemv(CblasColMajor, CblasTrans, n, jb,
                1.0, Xblk, n, v.data(), 1, 0.0, out.data() + j0, 1);
#else
    // Hand rolled: out_j = dot(X[:, j], v)
    // Good locality: we sweep columns in order; each column is contiguous in memory.
    for (int j = 0; j < jb; ++j) {
      const double* x = Xblk + std::ptrdiff_t(j) * n;
      // ddot
      double s = 0.0;
      for (int i = 0; i < n; ++i) s += x[i] * v[i];
      out[j0 + j] = s;
    }
#endif
  }
}

// y = X * v   (y size n)
static inline void gemv_X_vec(const MapMatC& X_, const Vec& v, Vec& y) {
  const int n = (int)X_.rows();
  const int p = (int)X_.cols();
  y.setZero(n);

  const int B = vd_colblock();
  for (int j0 = 0; j0 < p; j0 += B) {
    const int jb = std::min(B, p - j0);
    const double* Xblk = X_.col(j0).data(); // (n×jb)

#if defined(USE_CBLAS)
    // y += Xblk * v[j0:j0+jb]
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, jb,
                1.0, Xblk, n, v.data() + j0, 1, 1.0, y.data(), 1);
#else
    // y_i += sum_j X(i,j0+j) * v[j0+j]
    for (int j = 0; j < jb; ++j) {
      const double* x = Xblk + std::ptrdiff_t(j) * n;
      const double w = v[j0 + j];
      if (w == 0.0) continue;
      for (int i = 0; i < n; ++i) y[i] += x[i] * w;
    }
#endif
  }
}


// ---------- ctors ----------
VD_LARS::VD_LARS(const double* Xptr, int n, int p,
                 const double* yptr, int ny,
                 int num_dummies,
                 const VDOptions& o)
: n_(n), p_(p), L_(num_dummies),
  opt_(o), rng_(opt_.seed),
  X_(Xptr, n, p),   // Map constructed in initializer list (zero-copy)
  y_(yptr, ny)
{
  init_state_();
}

// Construct from Eigen matrices (zero-copy)
VD_LARS::VD_LARS(const Eigen::Ref<const MatC>& X_in,
                 const Eigen::Ref<const Vec>&  y_in,
                 int num_dummies,
                 const VDOptions& o)
: VD_LARS(X_in.data(), (int)X_in.rows(), (int)X_in.cols(),
          y_in.data(), (int)y_in.size(),
          num_dummies, o)
{}

// ---------- init ----------
// Initialize state variables
// ---------- init ----------
// Initialize all algorithm state and preallocate working buffers.
void VD_LARS::init_state_() {
  // Problem-scale helpers
  const int min_np = std::min(n_, p_);

  // Global flags/scalars
  standardized_ = false;
  y_norm_       = y_.norm();
  
  // (Optional) per-feature scaling kept as ones (if you later divide betas by it)
  normx_ = Vec::Ones(p_);

  // Model state
  beta_        = Vec::Zero(p_);
  beta_dummy_  = Vec::Zero(opt_.T_max);
  mu_          = Vec::Zero(n_);

  // y_ is already a Map<Vec> — keep a *copy* for mutable residuals (no re-map)
  residuals_   = y_;

  // Initial correlations: corr = X^T * residuals
  corr_.resize(p_);
  gemv_Xt_vec(X_, residuals_, corr_);  // uses CBLAS if available

  // Active sets & signs
  signs_.resize(0);                        // empty; will grow with actives
  signs_dummy_ = Vec::Zero(opt_.T_max);

  // Orthonormal basis (Q) and active block
  basis_.resize(n_, min_np);               // capacity; logical size tracked by basis_indices_
  X_active_.resize(n_, min_np);            // capacity for active columns

  // Activity bookkeeping
  is_active_.assign(p_, 0);
  pos_of_.assign(p_, -1);
  actives_.clear();                
  actives_.reserve(min_np);
  active_features_.clear();        
  active_features_.reserve(min_np);
  actives_set_.clear();

  // Reusable buffers
  proj_coeffs_.resize(std::max(1, min_np));
  ortho_buffer_.resize(n_);

  // Basis column tags: -1 = y, -2 = realized dummy, >=0 = real feature j
  basis_indices_.clear();
  basis_indices_.reserve(min_np);

  // Cholesky factor (empty)
  chol_factor_.resize(0, 0);

  // Realized dummies store
  X_realized_.setZero(n_, opt_.T_max);   // alloc + zero
  corr_realized_.setZero(opt_.T_max);
  T_realized_ = 0;

  // Virtual-dummy pool
  vd_stick_ = Vec::Ones(L_);
  vd_corr_  = Vec::Zero(L_);
  vd_is_realized_.assign(L_, 0);
  vd_unrealized_idx_.clear();            
  vd_unrealized_idx_.reserve(L_);

  // Projections matrix (rows added on demand; leave data uninitialized)
  vd_rows_cap_    = std::min(opt_.max_vd_proj, std::max(8, 128));
  vd_proj_.resize(vd_rows_cap_, L_);
  vd_rows_filled_ = 0;

  // Seed basis with normalized y, then sample/init VD row 0
  initialize_basis_();
  initialize_virtual_dummies_();
}

// Initialize basis with normalized y
void VD_LARS::initialize_basis_() {
  Eigen::Map<const Vec> yv(y_.data(), n_);
  const double yn = yv.norm();
  if (yn > opt_.eps * std::sqrt(double(n_))) {
    const int col = basis_size_();
    basis_.col(col) = yv / yn;
    basis_indices_.push_back(VD_LARS::Y_SENTINEL);  // sentinel for y
  }
}

// Return active indices as Eigen vector
Eigen::VectorXi VD_LARS::actives_indices_() const {
  Eigen::VectorXi out(int(actives_.size()));
  for (int i=0;i<(int)actives_.size();++i) out(i)=actives_[i];
  return out;
}

// Gram–Schmidt orthonormalization of v against current basis
std::optional<Vec> VD_LARS::orthonormalize_(const Vec& v) const {
  const int nb = basis_size_();
  if (nb == 0) {
    const double nv = v.norm();
    if (nv <= opt_.eps * std::sqrt(double(n_))) return std::nullopt;
    Vec u = v / nv;
    return u;
  }

  proj_coeffs_.head(nb).noalias() = basis_.leftCols(nb).transpose() * v;
  ortho_buffer_.noalias() = v - basis_.leftCols(nb) * proj_coeffs_.head(nb);

  const double nu = ortho_buffer_.norm();
  if (nu <= opt_.eps * std::sqrt(double(n_))) return std::nullopt;

  Vec out = ortho_buffer_ / nu;
  return out;
}

Vec VD_LARS::project_to_Vperp_(const Vec& z) const {
  const int nb = basis_size_();
  if (nb == 0) return z; // already in H; nothing to drop
  // coefficients in the current basis
  Vec coeff = basis_.leftCols(nb).transpose() * z;
  // subtract V-part; DO NOT normalize
  return z - basis_.leftCols(nb) * coeff;
}

// ---------- Marsaglia–Tsang method for Gamma(m,1) ----------
double VD_LARS::gamma_mt_(double m, std::mt19937_64& rng) {
    auto uniform01 = [&](){
        constexpr int mantissa_bits = 53;
        constexpr double scale = 1.0 / (1ULL << mantissa_bits);
        uint64_t x = rng() >> (64 - mantissa_bits);
        return x * scale;
    };
    std::normal_distribution<double> normal(0.0, 1.0);

    if (m < 1.0) {
        const double g = gamma_mt_(m + 1.0, rng);
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

// ---------- Virtual dummies ----------
// Initialize the first row of vd_proj_ and vd_stick_
void VD_LARS::initialize_virtual_dummies_() {
  ensure_vd_rows_capacity_(1);

  std::fill(vd_is_realized_.begin(), vd_is_realized_.end(), 0);
  vd_unrealized_idx_.clear();
  for (int d=0; d<L_; ++d) vd_unrealized_idx_.push_back(d);

  vd_rows_filled_ = 1; // we always fill at least one row

  if (opt_.dummy_law == DummyLaw::Spherical) {
    // ---- Base law (S): uniform on sphere via Beta(½, b) ----
    const double b = std::max(0.5 * double(n_ - 2), 1e-8);
    std::normal_distribution<double> N01(0.0, 1.0);
    double* row0 = vd_proj_.row(0).data();
    for (int d=0; d<L_; ++d) {
      const double z = N01(rng_);
      const double X = 0.5 * z * z;           // Gamma(1/2,1)
      const double Y = gamma_mt_(b, rng_);    // Gamma(b,1)
      const double U = X / (X + Y);           // Beta(1/2,b)
      const double sgn = (rng_() & 1ULL) ? 1.0 : -1.0;
      row0[d] = sgn * std::sqrt(U);
      vd_stick_(d) = 1.0 - row0[d] * row0[d];
    }
  } else {
    // ---- Base law (G): isotropic Gaussian projections ----
    std::normal_distribution<double> N01(0.0, 1.0);
    double scale = 1.0 / std::sqrt(double(n_));
    double* row0 = vd_proj_.row(0).data();
    for (int d=0; d<L_; ++d) {
        row0[d] = scale * N01(rng_);
        vd_stick_(d) = 1.0;  // unused in Gaussian case
    }
  }

  vd_corr_ = y_norm_ * vd_proj_.row(0).transpose();
}

// Helper to ensure vd_proj_ has enough rows
void VD_LARS::ensure_vd_rows_capacity_(int need) {
  if (need <= vd_rows_cap_) return;
  int new_cap = std::min(opt_.max_vd_proj,
                         std::max(vd_rows_cap_>0 ? vd_rows_cap_*2 : 128, need));
  MatR tmp(new_cap, L_);
  if (vd_rows_filled_>0) tmp.topRows(vd_rows_filled_) = vd_proj_.topRows(vd_rows_filled_);
  vd_proj_.swap(tmp);
  vd_rows_cap_ = new_cap;
}

// Update vd_proj_ with a new row if needed
void VD_LARS::update_virtual_dummies_() {
  const int m = basis_size_() - 1;
  if (m < 0 || m >= opt_.max_vd_proj) return;

  ensure_vd_rows_capacity_(m + 1);
  if (opt_.dummy_law == DummyLaw::Gaussian) {
    std::normal_distribution<double> N01(0.0, 1.0);
    double scale = 1.0 / std::sqrt(double(n_));
    double* row_m = vd_proj_.row(m).data();
    for (int d=0; d<L_; ++d) {
        if (!vd_is_realized_[d]) {
            row_m[d] = scale * N01(rng_);
        }
    }
    if (vd_rows_filled_ < m + 1) vd_rows_filled_ = m + 1;
    return;
  } // else DummyLaw::Spherical
  const double b = std::max(0.5 * double(n_ - m - 2), 1e-8);

  if ((int)vd_unrealized_idx_.size() == 0 || vd_rows_filled_ == 0) {
    vd_unrealized_idx_.clear();
    for (int d = 0; d < L_; ++d) {
      if (!vd_is_realized_[d] && vd_stick_(d) > opt_.eps) vd_unrealized_idx_.push_back(d);
    }
  }

  if (vd_unrealized_idx_.empty()) {
    if (vd_rows_filled_ < m + 1) vd_rows_filled_ = m + 1;
    return;
  }

  std::normal_distribution<double> N01(0.0, 1.0);
  double* row_m = vd_proj_.row(m).data();

  // update existing rows
  int keep = 0;
  for (int i = 0; i < (int)vd_unrealized_idx_.size(); ++i) {
    const int d = vd_unrealized_idx_[i];

    const double z  = N01(rng_);                      // N(0,1)
    const double X  = 0.5 * z * z;                    // Gamma(1/2,1)
    const double Y  = gamma_mt_(b, rng_);             // Gamma(b,1)
    const double U  = X / (X + Y);                    // Beta(1/2, b)
    const double sgn = (rng_() & 1ULL) ? 1.0 : -1.0;  // Rademacher

    const double st = vd_stick_(d);
    const double a  = sgn * std::sqrt(st * U);

    row_m[d]     = a;
    const double st_new = st - a * a;
    vd_stick_(d) = (st_new > 0.0 ? st_new : 0.0);

    if (!vd_is_realized_[d] && vd_stick_(d) > opt_.eps) {
      vd_unrealized_idx_[keep++] = d;
    }
  }
  if (keep != (int)vd_unrealized_idx_.size())
    vd_unrealized_idx_.resize(keep);

  if (vd_rows_filled_ < m + 1) vd_rows_filled_ = m + 1;
}


// Realize the vd_idx-th virtual dummy and store in X_realized_
void VD_LARS::realize_dummy_(int vd_idx) {
  const int j = T_realized_;

  const int nb = basis_size_();
  if (nb == 0) return;
  if (vd_proj_.rows() < nb) return;

  Eigen::VectorXd alphas = vd_proj_.topRows(nb).col(vd_idx);
  Vec explained(n_);
  explained.noalias() = basis_.leftCols(nb) * alphas;

  double stick = vd_stick_(vd_idx);
  if (stick < 0.0) stick = 0.0;
  const double r = std::sqrt(stick);

  std::normal_distribution<double> N01(0.0, 1.0);
  Eigen::VectorXd v(n_);
  for (int i = 0; i < n_; ++i) v(i) = N01(rng_);

  double m = v.mean();
  v.array() -= m;

  Vec x;
  if (opt_.dummy_law == DummyLaw::Spherical) {
    // Spherical: x = explained + r * orthonormalize(v)
    auto u_opt = orthonormalize_(v);
    if (!u_opt) return;
    x = explained + r * (*u_opt);
  } else {
    // Gaussian: x = explained + project_to_Vperp(scale * v)
    // No orthonormalization needed — project_to_Vperp_ subtracts the
    // V_k component without normalizing, preserving the Gaussian law.
    const double scale = 1.0 / std::sqrt(double(n_));
    Vec g_perp = project_to_Vperp_(scale * v);
    if (g_perp.norm() < opt_.eps * std::sqrt(double(n_))) return;
    x = explained + g_perp;
  }

  if (X_realized_.cols() <= j) {
    X_realized_.conservativeResize(Eigen::NoChange, j + 1);
  }
  X_realized_.col(j) = x;

  if (auto vo = orthonormalize_(x)) {
    const int col = basis_size_();
    basis_.col(col) = *vo;
    basis_indices_.push_back(VD_LARS::DUMMY_SENTINEL);   // realized dummy marker
    update_virtual_dummies_();
  }

  if (corr_realized_.size() < j + 1) corr_realized_.conservativeResize(j + 1);
  corr_realized_(j) = x.dot(residuals_);
  const double sgn = (corr_realized_(j) >= 0.0) ? 1.0 : -1.0;

  if (signs_dummy_.size() < j + 1) signs_dummy_.conservativeResize(j + 1);
  signs_dummy_(j) = sgn;

  // pool bookkeeping
  if (vd_idx >= 0 && vd_idx < vd_corr_.size()) vd_corr_(vd_idx) = 0.0;
  if (vd_idx >= 0 && vd_idx < (int)vd_is_realized_.size()) vd_is_realized_[vd_idx] = 1;
  vd_stick_(vd_idx) = 0.0;

  ++T_realized_;
}

// ---------- Core steps ----------
// Find and add new active features (real or VD)
// Returns the current max |corr| (C), or 0.0 if none found
double VD_LARS::find_and_add_active_() {
  // --- maxima over pools ---
  const double C_real = (p_ > 0) ? corr_.cwiseAbs().maxCoeff() : 0.0;

  double C_vd = 0.0;
  bool any_unrealized = false;
  if (L_ > 0) {
    // abs(vd_corr), mask realized with -inf so they can't win
    Eigen::ArrayXd abs_vd = vd_corr_.cwiseAbs().array();
    for (int d = 0; d < L_; ++d) {
      if (vd_is_realized_[d]) {
        abs_vd[d] = -std::numeric_limits<double>::infinity();
      } else {
        any_unrealized = true;
      }
    }
    if (any_unrealized) C_vd = abs_vd.maxCoeff();
  }

  const double C_rd = (T_realized_ > 0)
      ? corr_realized_.head(T_realized_).cwiseAbs().maxCoeff()
      : 0.0;

  const double C = std::max(C_real, std::max(C_vd, C_rd));
  if (C < 100.0 * opt_.eps) return 0.0;

  // --- VD wins before any reals are active ---
  if (active_features_.empty() && any_unrealized &&
      C_vd >= C_real - opt_.eps && C_vd >= C_rd - opt_.eps) {

    int best = -1;
    double bestv = -std::numeric_limits<double>::infinity();
    for (int d = 0; d < L_; ++d) if (!vd_is_realized_[d]) {
      const double v = std::abs(vd_corr_(d));
      if (v > bestv) { bestv = v; best = d; }
    }
    if (best >= 0) {
      realize_dummy_(best);
      const int jslot = T_realized_ - 1;
      active_features_.emplace_back(ActiveFeature{ActiveFeature::Kind::Dummy, jslot});
    }
    return C;
  }

  // --- add all reals with |corr| >= C - eps that are not active yet ---
  std::vector<int> new_actives;
  new_actives.reserve(16);
  const double thresh = C - opt_.eps;
  for (int j = 0; j < p_; ++j) {
    if (!is_active_[j] && std::abs(corr_(j)) >= thresh) {
      new_actives.push_back(j);
    }
  }

  // register and append to active_features_
  for (int j : new_actives) {
    actives_.push_back(j);
    actives_set_.insert(j);
    is_active_[j] = 1;
    pos_of_[j] = (int)actives_.size() - 1;
    active_features_.emplace_back(ActiveFeature{ActiveFeature::Kind::Real, j});
  }

  // signs = sign(corr[actives])
  if (!new_actives.empty()) {
    signs_.resize((int)actives_.size());
    for (int idx = 0; idx < (int)actives_.size(); ++idx) {
      const int j = actives_[idx];
      signs_(idx) = (corr_(j) >= 0.0) ? 1.0 : -1.0;
    }

    // Gram–Schmidt: add new signed columns to basis Q
    for (int j : new_actives) {
      const int pos = pos_of_[j];
      const double s = signs_(pos);
      Vec v = X_.col(j) * s;
      if (auto v_ortho = orthonormalize_(v)) {
        const int col = basis_size_();
        basis_.col(col) = *v_ortho;
        basis_indices_.push_back(j);
        update_virtual_dummies_();
      }
    }
  }

  return C;
}

// Update Cholesky factor with the last active feature
bool VD_LARS::chol_append(MatC& R,
                          const Eigen::Ref<const Vec>& v, // len t
                          double s,
                          double eps)

{
  const int t = R.rows();
  if (t == 0) {
    // initialize with a floor
    const double s0 = (s > eps ? s : eps);
    R.resize(1,1);
    R(0,0) = std::sqrt(s0);
    return true;
  }

  Eigen::VectorXd z = v;
  R.transpose().template triangularView<Eigen::Lower>().solveInPlace(z);

  double r2 = s - z.squaredNorm();

  // ---- TLARS-style flooring (no throw, no return false) ----
  // machine-precision-like positive floor, scaled by s to be safe
  const double scale_tol = std::max(eps, 1e-14 * std::max(1.0, s));
  if (r2 <= scale_tol) r2 = scale_tol;

  const double r = std::sqrt(r2);

  R.conservativeResize(t+1, t+1);
  R.topRightCorner(t,1) = z;
  R.row(t).setZero();
  R(t,t) = r;
  return true;
}

void VD_LARS::update_factor_() {
  if (active_features_.empty()) return;

  const int have = chol_factor_.rows();
  const int want = (int)active_features_.size();
  if (have >= want) return;

  const auto& af = active_features_[want - 1];
  const int idx  = af.index;

  // Build signed new column and stash into X_active_
  Vec x_new;
  if (af.kind == ActiveFeature::Kind::Real) {
    const int pos = pos_of_[idx];
    const double s = signs_(pos);
    x_new = X_.col(idx) * s;
  } else {
    x_new = X_realized_.col(idx) * signs_dummy_(idx);
  }
  X_active_.col(have) = x_new;

  if (have == 0) {
    double s = x_new.squaredNorm();
    // --- robust guard instead of throwing ---
    if (s <= opt_.eps) {
      // normalize if possible, else inject a unit-norm direction
      const double nrm = x_new.norm();
      if (nrm > opt_.eps) {
        X_active_.col(0) = x_new / nrm;
        s = 1.0;
      } else {
        // fall back to the first basis vector if available, or a random unit vector
        if (basis_size_() > 0) {
          X_active_.col(0) = basis_.col(0);
        } else {
          // make a deterministic safe vector
          X_active_.col(0).setZero();
          const int put = 0;
          X_active_.col(0)(put) = 1.0;
        }
        s = 1.0;
      }
    }
    chol_factor_.resize(1,1);
    chol_factor_(0,0) = std::sqrt(s);
    return;
  }

  // standard append
  Eigen::VectorXd v(have);
  v.noalias() = X_active_.leftCols(have).transpose() * x_new;
  const double s = x_new.squaredNorm();
  if (!chol_append(chol_factor_, v, s, opt_.eps)) {
    // soften: try tiny ridge instead of throwing
    MatC R = chol_factor_;
    if (!chol_append(R, v, s + 10.0 * opt_.eps, opt_.eps)) {
      // as a last resort, recompute from scratch for numerical safety
      MatC G = X_active_.leftCols(have+1).transpose() * X_active_.leftCols(have+1);
      Eigen::LLT<MatC> llt(G);
      if (llt.info() == Eigen::Success) {
        chol_factor_ = llt.matrixU();
      } else {
        // do not crash the worker; just keep the previous factor
        return;
      }
    } else {
      chol_factor_.swap(R);
    }
  }
}

// Compute direction components (w, u, a, a_vd, a_rd)
VD_LARS::Direction VD_LARS::compute_direction_() {
  if (chol_factor_.rows() == 0) throw std::runtime_error("chol_factor not initialized");
  const int nb = basis_size_();
  const int m = chol_factor_.rows();
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(m);
  Eigen::VectorXd z = ones;
  chol_factor_.transpose().template triangularView<Eigen::Lower>().solveInPlace(z);
  Eigen::VectorXd w0 = z;
  chol_factor_.template triangularView<Eigen::Upper>().solveInPlace(w0);

  const double denom = ones.dot(w0);
  const double A_act = 1.0 / std::sqrt(std::max(denom, opt_.eps));
  Eigen::VectorXd w  = A_act * w0;

  Vec u(n_);
  u.noalias() = X_active_.leftCols(m) * w;

  Vec a(p_);
#if defined(USE_CBLAS)
  cblas_dgemv(CblasColMajor, CblasTrans, n_, p_, 1.0, X_.data(), n_,
              u.data(), 1, 0.0, a.data(), 1);
#else
  a.noalias() = X_.transpose() * u;
#endif

  Eigen::VectorXd u_prime(nb);
  if (nb > 0) u_prime.noalias() = basis_.leftCols(nb).transpose() * u;

  const int m_rows = std::min(nb, vd_rows_filled_);
  Eigen::VectorXd a_vd;
  if (m_rows > 0) {
    // vd_proj_ is RowMajor, so topRows(m_rows).transpose() is a clean (L_ x m_rows) view
    a_vd = vd_proj_.topRows(m_rows).transpose() * u_prime.head(m_rows);
  } else {
    a_vd.resize(L_);
    a_vd.setZero();
  }

  Eigen::VectorXd a_rd;
  if (T_realized_ > 0) {
    a_rd = X_realized_.leftCols(T_realized_).transpose() * u;
  } else {
    a_rd.resize(0);
  }

  Direction out;
  out.A_active = A_act;
  out.w  = std::move(w);
  out.u  = std::move(u);
  out.a  = std::move(a);
  out.a_vd = std::move(a_vd);
  out.a_rd = std::move(a_rd);
  return out;
}

// Take a step of size gamma in direction (w,u), updating state
// Returns gamma, or 0.0 if no step taken
double VD_LARS::take_step_(double C, double A_active,
                           const Vec& w, const Vec& u,
                           const Vec& a, const Vec& a_vd, const Vec& a_rd)
{
  const double tol = opt_.eps; // or 10.0 * opt_.eps

  // build directions
  Vec d_real  = Vec::Zero(p_);
  Vec d_dummy = Vec::Zero(T_realized_);

  int w_idx = 0;
  for (const auto& af : active_features_) {
    const int idx = af.index;
    const double w_m = w(w_idx++);
    if (af.kind == ActiveFeature::Kind::Real) {
      const int pos = pos_of_[idx];
      const double s = signs_(pos);
      d_real(idx) = s * w_m;
    } else {
      if (idx < T_realized_) d_dummy(idx) = signs_dummy_(idx) * w_m;
    }
  }

  // ---- gammas for REAL pool (inactive only)
  double gamma_real = std::numeric_limits<double>::infinity();
  int    idx_real   = -1;

  for (int j = 0; j < p_; ++j) {
    if (is_active_[j]) continue;
    const double aj = a(j), cj = corr_(j);

    const double den1 = A_active - aj;
    if (den1 > tol) {
      const double t = (C - cj) / den1;
      if (t > tol && t < gamma_real) { gamma_real = t; idx_real = j; }
    }

    const double den2 = A_active + aj;
    if (den2 > tol) {
      const double t = (C + cj) / den2;
      if (t > tol && t < gamma_real) { gamma_real = t; idx_real = j; }
    }
  }

  // ---- gammas for VD pool (unrealized only)
  double gamma_vd = std::numeric_limits<double>::infinity();
  int    idx_vd   = -1;

  for (int d = 0; d < L_; ++d) {
    if (vd_is_realized_[d]) continue;
    const double aj = a_vd(d);
    const double cj = vd_corr_(d);

    const double den1 = A_active - aj;
    if (den1 > tol) {
      const double t = (C - cj) / den1;
      if (t > tol && t < gamma_vd) { gamma_vd = t; idx_vd = d; }
    }

    const double den2 = A_active + aj;
    if (den2 > tol) {
      const double t = (C + cj) / den2;
      if (t > tol && t < gamma_vd) { gamma_vd = t; idx_vd = d; }
    }
  }

  // pick winner (deterministic tie-break)
  bool winner_is_vd = (gamma_vd + tol < gamma_real);
  double gamma = winner_is_vd ? gamma_vd : gamma_real;

  if (!(gamma > opt_.eps)) return 0.0;

  // state updates
  mu_.noalias()   += gamma * u;
  beta_.noalias() += gamma * d_real;
  if (T_realized_ > 0 && d_dummy.size() == T_realized_) {
    beta_dummy_.head(T_realized_).noalias() += gamma * d_dummy;
  }

  residuals_ = y_ - mu_;
  corr_.noalias() -= gamma * a;

  for (int d = 0; d < L_; ++d) {
    if (!vd_is_realized_[d]) {
      vd_corr_(d) -= gamma * a_vd(d);
    }
  }

  if (T_realized_ > 0 && a_rd.size() == T_realized_) {
    corr_realized_.head(T_realized_).noalias() -= gamma * a_rd;
  }

  // realize VD if it won the step
  if (winner_is_vd && idx_vd >= 0) {
    realize_dummy_(idx_vd);
    int jslot = T_realized_ - 1;
    active_features_.emplace_back(ActiveFeature{ActiveFeature::Kind::Dummy, jslot});
  }

  return gamma;
}

// ---------- public run ----------
// Run the algorithm until at least T dummies are realized, or max steps reached
// Returns the coefficient path (p x #steps)
MatC VD_LARS::run(int T) {
  std::vector<Vec> path;
  auto record_beta_scaled = [&](){
    Vec scaled = beta_;
    if (normx_.size() == beta_.size()) {
      scaled.array() /= normx_.array();
    }
    path.emplace_back(std::move(scaled));
  };

  if (step_ == 0) {
    record_beta_scaled();
  }

  const int max_steps = p_ + opt_.T_max;

  for (int it = step_; it < max_steps; ++it) {
    const int prev_realized = T_realized_;

    double C = find_and_add_active_();
    if (C <= 0.0) break;
    update_factor_();
    Direction dir = compute_direction_();
    double gamma = take_step_(C, dir.A_active, dir.w, dir.u, dir.a, dir.a_vd, dir.a_rd);
    if (gamma <= opt_.eps) break;
    record_beta_scaled();

    if (T_realized_ > prev_realized && T_realized_ >= T) {
      step_ = it + 1;
      break;
    }
    step_ = it + 1;
  }

  const int cols = static_cast<int>(path.size());
  MatC out(p_, std::max(cols, 1));
  if (cols == 0) {
    out.col(0).setZero();
    return out;
  }
  for (int m = 0; m < cols; ++m) out.col(m) = path[m];
  return out;
}