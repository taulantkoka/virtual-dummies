// vd_omp.cpp
// Virtual Dummy Orthogonal Matching Pursuit.
#include "vd_omp.hpp"
using DummyLaw = VDDummyLaw;

// ========================================================================
// GEMV helpers — same streaming-block pattern as VD-LARS
// ========================================================================
namespace {

using vd_detail::colblock;

// out = X^T * v   (size p)
inline void gemv_Xt_vec(const MapMatC& X, const Vec& v, Vec& out) {
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

// y = X * v   (size n)
inline void gemv_X_vec(const MapMatC& X, const Vec& v, Vec& y) {
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

// Cholesky append: grow R such that R^T R = G_{k+1}.
// v = old X_A^T x_new  (length t),  s = x_new^T x_new.
bool chol_append(MatC& R,
                 const Eigen::Ref<const Vec>& v,
                 double s,
                 double eps)
{
  const int t = R.rows();
  if (t == 0) {
    const double s0 = (s > eps ? s : eps);
    R.resize(1, 1);
    R(0, 0) = std::sqrt(s0);
    return true;
  }

  Eigen::VectorXd z = v;
  R.transpose().template triangularView<Eigen::Lower>().solveInPlace(z);

  double r2 = s - z.squaredNorm();
  const double floor = std::max(eps, 1e-14 * std::max(1.0, s));
  if (r2 <= floor) r2 = floor;

  R.conservativeResize(t + 1, t + 1);
  R.topRightCorner(t, 1) = z;
  R.row(t).setZero();
  R(t, t) = std::sqrt(r2);
  return true;
}

} // anonymous namespace

// ========================================================================
// Constructors
// ========================================================================
VD_OMP::VD_OMP(const double* Xptr, int n, int p,
               const double* yptr, int ny,
               int num_dummies,
               const VDOptions& o)
  : n_(n), p_(p), L_(num_dummies),
    opt_(o), rng_(opt_.seed),
    X_(Xptr, n, p),
    y_(yptr, ny)
{
  init_state_();
}

VD_OMP::VD_OMP(const Eigen::Ref<const MatC>& X_in,
               const Eigen::Ref<const Vec>&  y_in,
               int num_dummies,
               const VDOptions& o)
  : VD_OMP(X_in.data(), (int)X_in.rows(), (int)X_in.cols(),
           y_in.data(), (int)y_in.size(),
           num_dummies, o)
{}

// ========================================================================
// Initialization
// ========================================================================
void VD_OMP::init_state_() {
  const int min_np = std::min(n_, p_);

  standardized_ = false;
  y_norm_       = y_.norm();
  normx_        = Vec::Ones(p_);

  // Model state
  beta_       = Vec::Zero(p_);
  beta_dummy_ = Vec::Zero(opt_.T_max);
  mu_         = Vec::Zero(n_);
  residuals_  = y_;  // copy from map

  // Initial correlations: corr = X^T y
  corr_.resize(p_);
  gemv_Xt_vec(X_, residuals_, corr_);

  // Active set
  is_active_.assign(p_, 0);
  actives_.clear();
  actives_.reserve(min_np);
  active_features_.clear();
  active_features_.reserve(min_np);

  // Basis
  basis_.resize(n_, min_np);
  basis_indices_.clear();
  basis_indices_.reserve(min_np);

  // Reusable buffers
  proj_coeffs_.resize(std::max(1, min_np));
  ortho_buffer_.resize(n_);

  // Active columns, Cholesky, and X_A^T y
  X_active_.resize(n_, min_np);
  chol_factor_.resize(0, 0);
  Xty_active_.resize(0);

  // Realized dummies
  X_realized_.setZero(n_, opt_.T_max);
  corr_realized_.setZero(opt_.T_max);
  T_realized_ = 0;

  // Virtual-dummy pool
  vd_stick_ = Vec::Ones(L_);
  vd_corr_  = Vec::Zero(L_);
  vd_is_realized_.assign(L_, 0);
  vd_unrealized_idx_.clear();
  vd_unrealized_idx_.reserve(L_);

  vd_rows_cap_    = std::min(opt_.max_vd_proj, std::max(8, 128));
  vd_proj_.resize(vd_rows_cap_, L_);
  vd_rows_filled_ = 0;

  // Seed basis with normalized y, then init VD
  initialize_basis_();
  initialize_virtual_dummies_();
}

void VD_OMP::initialize_basis_() {
  Eigen::Map<const Vec> yv(y_.data(), n_);
  const double yn = yv.norm();
  if (yn > opt_.eps * std::sqrt(double(n_))) {
    basis_.col(0) = yv / yn;
    basis_indices_.push_back(VD_Y_SENTINEL);
  }
}

Eigen::VectorXi VD_OMP::active_indices() const {
  Eigen::VectorXi out(int(actives_.size()));
  for (int i = 0; i < (int)actives_.size(); ++i) out(i) = actives_[i];
  return out;
}

// ========================================================================
// Basis helpers (identical to VD-LARS)
// ========================================================================
std::optional<Vec> VD_OMP::orthonormalize_(const Vec& v) const {
  const int nb = basis_size_();
  if (nb == 0) {
    const double nv = v.norm();
    if (nv <= opt_.eps * std::sqrt(double(n_))) return std::nullopt;
    return Vec(v / nv);
  }
  proj_coeffs_.head(nb).noalias() = basis_.leftCols(nb).transpose() * v;
  ortho_buffer_.noalias() = v - basis_.leftCols(nb) * proj_coeffs_.head(nb);
  const double nu = ortho_buffer_.norm();
  if (nu <= opt_.eps * std::sqrt(double(n_))) return std::nullopt;
  return Vec(ortho_buffer_ / nu);
}

Vec VD_OMP::project_to_Vperp_(const Vec& z) const {
  const int nb = basis_size_();
  if (nb == 0) return z;
  Vec coeff = basis_.leftCols(nb).transpose() * z;
  return z - basis_.leftCols(nb) * coeff;
}

// ========================================================================
// Virtual-dummy machinery (identical to VD-LARS)
// ========================================================================
void VD_OMP::ensure_vd_rows_capacity_(int need) {
  if (need <= vd_rows_cap_) return;
  int new_cap = std::min(opt_.max_vd_proj,
                         std::max(vd_rows_cap_ > 0 ? vd_rows_cap_ * 2 : 128, need));
  MatR tmp(new_cap, L_);
  if (vd_rows_filled_ > 0)
    tmp.topRows(vd_rows_filled_) = vd_proj_.topRows(vd_rows_filled_);
  vd_proj_.swap(tmp);
  vd_rows_cap_ = new_cap;
}

void VD_OMP::initialize_virtual_dummies_() {
  ensure_vd_rows_capacity_(1);
  std::fill(vd_is_realized_.begin(), vd_is_realized_.end(), 0);
  vd_unrealized_idx_.clear();
  for (int d = 0; d < L_; ++d) vd_unrealized_idx_.push_back(d);
  vd_rows_filled_ = 1;

  if (opt_.dummy_law == DummyLaw::Spherical) {
    const double b = std::max(0.5 * double(n_ - 2), 1e-8);
    std::normal_distribution<double> N01(0.0, 1.0);
    double* row0 = vd_proj_.row(0).data();
    for (int d = 0; d < L_; ++d) {
      const double z = N01(rng_);
      const double X = 0.5 * z * z;
      const double Y = vd_detail::gamma_mt(b, rng_);
      const double U = X / (X + Y);
      const double sgn = (rng_() & 1ULL) ? 1.0 : -1.0;
      row0[d] = sgn * std::sqrt(U);
      vd_stick_(d) = 1.0 - row0[d] * row0[d];
    }
  } else {
    std::normal_distribution<double> N01(0.0, 1.0);
    double scale = 1.0 / std::sqrt(double(n_));
    double* row0 = vd_proj_.row(0).data();
    for (int d = 0; d < L_; ++d) {
      row0[d] = scale * N01(rng_);
      vd_stick_(d) = 1.0;
    }
  }
  vd_corr_ = y_norm_ * vd_proj_.row(0).transpose();
}

void VD_OMP::update_virtual_dummies_() {
  const int m = basis_size_() - 1;
  if (m < 0 || m >= opt_.max_vd_proj) return;

  ensure_vd_rows_capacity_(m + 1);

  if (opt_.dummy_law == DummyLaw::Gaussian) {
    std::normal_distribution<double> N01(0.0, 1.0);
    double scale = 1.0 / std::sqrt(double(n_));
    double* row_m = vd_proj_.row(m).data();
    for (int d = 0; d < L_; ++d) {
      if (!vd_is_realized_[d]) row_m[d] = scale * N01(rng_);
    }
    if (vd_rows_filled_ < m + 1) vd_rows_filled_ = m + 1;
    return;
  }

  // Spherical stick-breaking
  const double b = std::max(0.5 * double(n_ - m - 2), 1e-8);

  if ((int)vd_unrealized_idx_.size() == 0 || vd_rows_filled_ == 0) {
    vd_unrealized_idx_.clear();
    for (int d = 0; d < L_; ++d) {
      if (!vd_is_realized_[d] && vd_stick_(d) > opt_.eps)
        vd_unrealized_idx_.push_back(d);
    }
  }
  if (vd_unrealized_idx_.empty()) {
    if (vd_rows_filled_ < m + 1) vd_rows_filled_ = m + 1;
    return;
  }

  std::normal_distribution<double> N01(0.0, 1.0);
  double* row_m = vd_proj_.row(m).data();

  int keep = 0;
  for (int i = 0; i < (int)vd_unrealized_idx_.size(); ++i) {
    const int d = vd_unrealized_idx_[i];
    const double z   = N01(rng_);
    const double X   = 0.5 * z * z;
    const double Y   = vd_detail::gamma_mt(b, rng_);
    const double U   = X / (X + Y);
    const double sgn = (rng_() & 1ULL) ? 1.0 : -1.0;
    const double st  = vd_stick_(d);
    const double a   = sgn * std::sqrt(st * U);
    row_m[d] = a;
    const double st_new = st - a * a;
    vd_stick_(d) = (st_new > 0.0 ? st_new : 0.0);
    if (!vd_is_realized_[d] && vd_stick_(d) > opt_.eps)
      vd_unrealized_idx_[keep++] = d;
  }
  if (keep != (int)vd_unrealized_idx_.size())
    vd_unrealized_idx_.resize(keep);
  if (vd_rows_filled_ < m + 1) vd_rows_filled_ = m + 1;
}

void VD_OMP::realize_dummy_(int vd_idx) {
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
  double mean = v.mean();
  v.array() -= mean;

  Vec x;
  if (opt_.dummy_law == DummyLaw::Spherical) {
    auto u_opt = orthonormalize_(v);
    if (!u_opt) return;
    x = explained + r * (*u_opt);
  } else {
    const double scale = 1.0 / std::sqrt(double(n_));
    Vec g_perp = project_to_Vperp_(scale * v);
    if (g_perp.norm() < opt_.eps * std::sqrt(double(n_))) return;
    x = explained + g_perp;
  }

  if (X_realized_.cols() <= j)
    X_realized_.conservativeResize(Eigen::NoChange, j + 1);
  X_realized_.col(j) = x;

  // Extend basis with orthonormalized realized dummy
  if (auto vo = orthonormalize_(x)) {
    const int col = basis_size_();
    basis_.col(col) = *vo;
    basis_indices_.push_back(VD_DUMMY_SENTINEL);
    update_virtual_dummies_();
  }

  if (corr_realized_.size() < j + 1) corr_realized_.conservativeResize(j + 1);
  corr_realized_(j) = x.dot(residuals_);

  // Pool bookkeeping
  if (vd_idx >= 0 && vd_idx < vd_corr_.size()) vd_corr_(vd_idx) = 0.0;
  if (vd_idx >= 0 && vd_idx < (int)vd_is_realized_.size()) vd_is_realized_[vd_idx] = 1;
  vd_stick_(vd_idx) = 0.0;

  ++T_realized_;
}

// ========================================================================
// OMP-specific: candidate selection
// ========================================================================
std::optional<VD_OMP::Candidate> VD_OMP::find_best_candidate_() const {
  Candidate best{Candidate::Pool::Real, -1, 0.0};

  // Scan inactive reals
  for (int j = 0; j < p_; ++j) {
    if (is_active_[j]) continue;
    const double ac = std::abs(corr_(j));
    if (ac > best.abs_corr) {
      best = {Candidate::Pool::Real, j, ac};
    }
  }

  // Scan unrealized virtual dummies
  for (int d = 0; d < L_; ++d) {
    if (vd_is_realized_[d]) continue;
    const double ac = std::abs(vd_corr_(d));
    if (ac > best.abs_corr) {
      best = {Candidate::Pool::VD, d, ac};
    }
  }

  // Realized-but-not-yet-active dummies should not exist in OMP
  // (every realized dummy is immediately added to the active set),
  // but scan for safety / diagnostic consistency.
  for (int j = 0; j < T_realized_; ++j) {
    // Check if this realized dummy is already active
    bool already_active = false;
    for (const auto& af : active_features_) {
      if (af.kind == ActiveFeature::Kind::Dummy && af.index == j) {
        already_active = true;
        break;
      }
    }
    if (already_active) continue;
    const double ac = std::abs(corr_realized_(j));
    if (ac > best.abs_corr) {
      best = {Candidate::Pool::Realized, j, ac};
    }
  }

  if (best.index < 0 || best.abs_corr < 100.0 * opt_.eps)
    return std::nullopt;
  return best;
}

// ========================================================================
// OMP-specific: append column to active set / Cholesky
// ========================================================================
void VD_OMP::append_to_factor_(const Vec& x_col) {
  const int k = (int)active_features_.size() - 1;  // new column index in X_active_
  X_active_.col(k) = x_col;

  // Extend X_A^T y
  Xty_active_.conservativeResize(k + 1);
  Xty_active_(k) = x_col.dot(y_);

  if (k == 0) {
    double s = x_col.squaredNorm();
    if (s <= opt_.eps) s = opt_.eps;
    chol_factor_.resize(1, 1);
    chol_factor_(0, 0) = std::sqrt(s);
    return;
  }

  // Cross-products with existing active columns
  Eigen::VectorXd v(k);
  v.noalias() = X_active_.leftCols(k).transpose() * x_col;
  const double s = x_col.squaredNorm();

  if (!chol_append(chol_factor_, v, s, opt_.eps)) {
    // Fallback: recompute from scratch
    MatC G = X_active_.leftCols(k + 1).transpose() * X_active_.leftCols(k + 1);
    Eigen::LLT<MatC> llt(G);
    if (llt.info() == Eigen::Success) {
      chol_factor_ = llt.matrixU();
    }
  }
}

// ========================================================================
// OMP-specific: OLS refit
// ========================================================================
void VD_OMP::ols_refit_() {
  const int k = (int)active_features_.size();
  if (k == 0) return;

  // Solve R^T z = X_A^T y, then R β_A = z
  Vec z = Xty_active_;
  chol_factor_.topLeftCorner(k, k)
    .transpose()
    .template triangularView<Eigen::Lower>()
    .solveInPlace(z);
  chol_factor_.topLeftCorner(k, k)
    .template triangularView<Eigen::Upper>()
    .solveInPlace(z);
  // z now holds β_A

  // Scatter into full β vectors
  beta_.setZero();
  beta_dummy_.head(std::max(T_realized_, 1)).setZero();

  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Real) {
      beta_(af.index) = z(i);
    } else {
      if (af.index < beta_dummy_.size())
        beta_dummy_(af.index) = z(i);
    }
  }

  // μ = X_A β_A
  mu_.noalias() = X_active_.leftCols(k) * z;

  // Residual
  residuals_ = y_ - mu_;

  // ---- Full correlation refresh ----
  // Real features: corr = X^T r
  gemv_Xt_vec(X_, residuals_, corr_);

  // Virtual dummies: vd_corr = A_k^T (E_k^T r)
  const int nb = basis_size_();
  const int m_rows = std::min(nb, vd_rows_filled_);
  if (m_rows > 0 && L_ > 0) {
    Vec basis_proj = basis_.leftCols(nb).transpose() * residuals_;
    vd_corr_.noalias() = vd_proj_.topRows(m_rows).transpose() * basis_proj.head(m_rows);
    // Zero out realized dummies
    for (int d = 0; d < L_; ++d) {
      if (vd_is_realized_[d]) vd_corr_(d) = 0.0;
    }
  }

  // Realized dummies
  if (T_realized_ > 0) {
    corr_realized_.head(T_realized_).noalias() =
      X_realized_.leftCols(T_realized_).transpose() * residuals_;
  }
}

// ========================================================================
// Public run
// ========================================================================
MatC VD_OMP::run(int T) {
  std::vector<Vec> path;
  auto record_beta = [&]() {
    Vec scaled = beta_;
    if (normx_.size() == beta_.size())
      scaled.array() /= normx_.array();
    path.emplace_back(std::move(scaled));
  };

  if (step_ == 0) record_beta();  // column 0 = zeros

  const int max_steps = std::min(n_, p_) + opt_.T_max;

  for (int it = step_; it < max_steps; ++it) {
    const int prev_realized = T_realized_;

    // 1. Find best inactive candidate
    auto cand_opt = find_best_candidate_();
    if (!cand_opt) break;
    const auto& cand = *cand_opt;

    // 2. Activate the winner
    Vec x_col;
    if (cand.pool == Candidate::Pool::VD) {
      // Realize the virtual dummy, then add it
      realize_dummy_(cand.index);
      const int jslot = T_realized_ - 1;
      active_features_.emplace_back(ActiveFeature{ActiveFeature::Kind::Dummy, jslot});
      x_col = X_realized_.col(jslot);
    } else if (cand.pool == Candidate::Pool::Real) {
      const int j = cand.index;
      actives_.push_back(j);
      is_active_[j] = 1;
      active_features_.emplace_back(ActiveFeature{ActiveFeature::Kind::Real, j});
      x_col = X_.col(j);

      // Extend basis with orthonormalized real column
      if (auto vo = orthonormalize_(x_col)) {
        const int col = basis_size_();
        basis_.col(col) = *vo;
        basis_indices_.push_back(j);
        update_virtual_dummies_();
      }
    } else {
      // Realized-but-not-active (shouldn't normally happen)
      const int jslot = cand.index;
      active_features_.emplace_back(ActiveFeature{ActiveFeature::Kind::Dummy, jslot});
      x_col = X_realized_.col(jslot);
    }

    // 3. Append to Cholesky factor
    append_to_factor_(x_col);

    // 4. OLS refit + full correlation refresh
    ols_refit_();

    record_beta();

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