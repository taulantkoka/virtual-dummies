// vd_afs.hpp
// Virtual Dummy Adaptive Forward Stepwise (VD-AFS).
//
// AFS (Zhang & Tibshirani, JMLR 2026) interpolates between Forward
// Stepwise (ρ=1) and LARS (ρ→0) via a shrinkage blend:
//
//   β^AFS_m = (1-ρ) β^AFS_{m-1} + ρ ν̂_m     ν̂_m = OLS on active set
//
// Unlike OMP, AFS scans *all* candidates at each step (including
// already-active features).  Re-selecting an active feature does not
// change the active set but applies another shrinkage blend, moving
// β^AFS closer to the OLS estimate.
//
// The selection rule φ_k = argmax_j |⟨x_j, r_k⟩| with
// r_k = y − X β^AFS_{k-1} is F_k-measurable (r_k ∈ V_k), so
// Theorem 1 applies and VD-AFS has the same law as augmented AFS.
//
// At ρ=1, VD-AFS reduces to VD-OMP (Forward Stepwise).
#pragma once
#include "vd_common.hpp"

class VD_AFS {
public:
  // ---------- Constructors ----------
  VD_AFS(const double* Xptr, int n, int p,
         const double* yptr, int ny,
         int num_dummies,
         const VDOptions& o);

  VD_AFS(const Eigen::Ref<const MatC>& X_in,
         const Eigen::Ref<const Vec>&  y_in,
         int num_dummies,
         const VDOptions& o);

  // ---------- Top-level API ----------
  MatC run(int T = 1);

  // ---------- Accessors ----------
  int num_dummies()          const noexcept { return L_; }
  int num_realized_dummies() const noexcept { return T_realized_; }
  int n_features()           const noexcept { return p_; }
  int n_samples()            const noexcept { return n_; }
  int basis_size()           const noexcept { return basis_size_(); }
  double rho()               const noexcept { return rho_; }

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

  struct ActiveFeature {
    enum class Kind : uint8_t { Real, Dummy };
    Kind kind;
    int  index;
  };

  static constexpr int Y_SENTINEL     = VD_Y_SENTINEL;
  static constexpr int DUMMY_SENTINEL = VD_DUMMY_SENTINEL;

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

private:
  // ---------- Problem sizes & options ----------
  int n_ = 0, p_ = 0, L_ = 0;
  double rho_ = 1.0;
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

  // Active set
  std::vector<int> actives_;
  std::vector<ActiveFeature> active_features_;
  std::vector<char> is_active_;

  // Basis
  MatC basis_;
  std::vector<int> basis_indices_;
  inline int basis_size_() const noexcept { return static_cast<int>(basis_indices_.size()); }

  // Reusable buffers
  mutable Vec proj_coeffs_;
  mutable Vec ortho_buffer_;

  // Active columns, Cholesky, X_A^T y, cached OLS solution
  MatC X_active_;
  MatC chol_factor_;
  Vec  Xty_active_;
  Vec  nu_active_;        // ν̂_m = OLS on active set (cached, recomputed on A growth)
  bool nu_stale_ = true;  // true if active set changed since last OLS solve

  // Realized dummies
  MatC X_realized_;
  Vec  corr_realized_;
  int  T_realized_ = 0;

  // Virtual-dummy pool
  MatR vd_proj_;
  Vec  vd_stick_, vd_corr_;
  std::vector<char> vd_is_realized_;
  std::vector<int>  vd_unrealized_idx_;
  int vd_rows_filled_ = 0, vd_rows_cap_ = 0;

  // Bookkeeping
  int step_ = 0;

  // ---------- Setup ----------
  void init_state_();
  void initialize_basis_();
  void ensure_vd_rows_capacity_(int need);

  // ---------- VD machinery ----------
  std::optional<Vec> orthonormalize_(const Vec& v) const;
  Vec  project_to_Vperp_(const Vec& z) const;
  void initialize_virtual_dummies_();
  void update_virtual_dummies_();
  void realize_dummy_(int vd_idx);

  // ---------- AFS-specific ----------
  struct Candidate {
    enum class Pool : uint8_t { Real, VD, RealizedDummy };
    Pool   pool;
    int    index;
    double abs_corr;
    bool   is_new;   // not yet in active set
  };
  std::optional<Candidate> find_best_candidate_() const;

  void append_to_factor_(const Vec& x_col);
  void ols_solve_();       // solve ν̂ = R^{-1} R^{-T} X_A^T y, cache in nu_active_
  void afs_blend_();       // β = (1-ρ) β + ρ ν̂, μ update, residual, correlation refresh

  VD_AFS(const VD_AFS&) = delete;
  VD_AFS& operator=(const VD_AFS&) = delete;
};