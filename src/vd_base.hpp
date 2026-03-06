// vd_base.hpp
// Base class for VD-* forward selectors.
// Owns all shared state: data views, VD pool, basis, Cholesky,
// model state, scratch buffers, and public accessors.
// Solvers (LARS, OMP, AFS) inherit and implement run().
#pragma once
#include "vd_common.hpp"

class VD_Base {
public:
  // ---------- Constructors ----------
  VD_Base(const double* Xptr, int n, int p,
          const double* yptr, int ny,
          int num_dummies, const VDOptions& o);

  VD_Base(const Eigen::Ref<const MatC>& X_in,
          const Eigen::Ref<const Vec>&  y_in,
          int num_dummies, const VDOptions& o);

  virtual ~VD_Base() = default;

  // Pure virtual: each solver implements its own run loop.
  virtual MatC run(int T = 1) = 0;

  // ---------- Public accessors ----------
  int num_dummies()          const noexcept { return L_; }
  int num_realized_dummies() const noexcept { return T_realized_; }
  int n_features()           const noexcept { return p_; }
  int n_samples()            const noexcept { return n_; }
  int basis_size()           const noexcept { return basis_size_(); }

  Eigen::VectorXd beta_view_copy()   const { return beta_; }
  Eigen::VectorXd corr_view_copy()   const { return corr_; }
  Eigen::VectorXi active_indices()   const;

  Eigen::VectorXd corr_realized_view_copy() const {
    return corr_realized_.head(T_realized_);
  }
  Eigen::VectorXi is_dummy_realized_view() const {
    Eigen::VectorXi out(L_);
    for (int d = 0; d < L_; ++d) out(d) = vd_is_realized_[d] ? 1 : 0;
    return out;
  }

  std::vector<ActiveFeature> active_features_copy() const { return active_features_; }

  Eigen::VectorXd beta_real() const {
    Eigen::VectorXd out(p_);
    if (beta_.size() >= p_) out = beta_.head(p_);
    else                    out.setZero();
    return out;
  }

  const Vec&  normx_view()    const noexcept { return normx_; }
  const MatR& vd_proj_view()  const noexcept { return vd_proj_; }
  const Vec&  vd_corr_view()  const noexcept { return vd_corr_; }
  const Vec&  vd_stick_view() const noexcept { return vd_stick_; }

protected:
  // ---------- Problem sizes & options ----------
  int n_ = 0, p_ = 0, L_ = 0;
  VDOptions opt_;

  // ---------- Data (mapped views) ----------
  MapMatC X_;
  MapVec  y_;

  // ---------- RNG ----------
  std::mt19937_64 rng_{0};

  // ---------- Global scalars ----------
  bool   standardized_ = false;
  double y_norm_ = 0.0;

  // ---------- Model state ----------
  Vec normx_;
  Vec beta_, beta_dummy_, mu_, residuals_, corr_;

  // ---------- Active set ----------
  std::vector<int> actives_;                    // real-feature indices
  std::vector<ActiveFeature> active_features_;  // all active (real + dummy)
  std::vector<char> is_active_;                 // O(1) lookup for reals

  // ---------- Basis ----------
  MatC basis_;
  std::vector<int> basis_indices_;
  inline int basis_size_() const noexcept {
    return static_cast<int>(basis_indices_.size());
  }

  // ---------- Reusable buffers ----------
  mutable Vec proj_coeffs_;
  mutable Vec ortho_buffer_;

  // ---------- Active columns & Cholesky ----------
  MatC X_active_;
  MatC chol_factor_;

  // ---------- Realized dummies ----------
  MatC X_realized_;
  Vec  corr_realized_;
  int  T_realized_ = 0;

  // ---------- Virtual-dummy pool ----------
  MatR vd_proj_;
  Vec  vd_stick_, vd_corr_;
  std::vector<char> vd_is_realized_;
  std::vector<int>  vd_unrealized_idx_;
  int vd_rows_filled_ = 0, vd_rows_cap_ = 0;

  // ---------- pread scratch ----------
  std::vector<double> gemv_scratch_;

  // ---------- Bookkeeping ----------
  int step_ = 0;

  // ---------- Shared setup ----------
  void init_common_state_();    // alloc + init everything above
  void initialize_basis_();
  void ensure_vd_rows_capacity_(int need);

  // ---------- Shared VD machinery ----------
  std::optional<Vec> orthonormalize_(const Vec& v) const;
  Vec  project_to_Vperp_(const Vec& z) const;
  void initialize_virtual_dummies_();
  void update_virtual_dummies_();
  void realize_dummy_(int vd_idx);

  // ---------- Shared Cholesky ----------
  static bool chol_append(MatC& R,
                          const Eigen::Ref<const Vec>& v,
                          double s, double eps);

  // ---------- GEMV convenience ----------
  double* scratch_ptr_() {
    return gemv_scratch_.empty() ? nullptr : gemv_scratch_.data();
  }
  void full_corr_refresh_();    // corr_ = X^T r, vd_corr, corr_realized

private:
  VD_Base(const VD_Base&) = delete;
  VD_Base& operator=(const VD_Base&) = delete;
};
