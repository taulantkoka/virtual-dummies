// vd_omp.hpp
// Virtual Dummy Orthogonal Matching Pursuit (VD-OMP).
//
// OMP is a greedy forward selector: at each step it picks the candidate
// (real or virtual dummy) with the largest absolute correlation with the
// current residual, adds it to the active set, and refits OLS on the
// active columns.  The VD machinery replaces the explicit dummy block
// D ∈ R^{n×L} with sequential projections, exactly as in VD-LARS.
//
// The selection rule φ_k = argmax |⟨x_j, r_k⟩| is F_k-measurable
// (Table 1 in the paper), so Theorem 1 applies and the augmented and
// virtual-dummy OMP paths have identical law.
#pragma once
#include "vd_common.hpp"

class VD_OMP {
public:
  // ---------- Constructors (X, y are mapped views; no copies) ----------
  VD_OMP(const double* Xptr, int n, int p,
         const double* yptr, int ny,
         int num_dummies,
         const VDOptions& o);

  VD_OMP(const Eigen::Ref<const MatC>& X_in,
         const Eigen::Ref<const Vec>&  y_in,
         int num_dummies,
         const VDOptions& o);

  // ---------- Top-level API ----------
  // Advance until at least T dummies are realized (or no progress).
  // Returns the coefficient path: p × (#steps+1), column 0 = zeros.
  MatC run(int T = 1);

  // ---------- Accessors ----------
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

  // Nested ActiveFeature (mirrors VD_LARS::ActiveFeature)
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

  // Read-only views into VD state
  const Vec&  normx_view()    const noexcept { return normx_; }
  const MatR& vd_proj_view()  const noexcept { return vd_proj_; }
  const Vec&  vd_corr_view()  const noexcept { return vd_corr_; }
  const Vec&  vd_stick_view() const noexcept { return vd_stick_; }

private:
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

  // Active set
  std::vector<int> actives_;             // real-feature indices in active set
  std::vector<ActiveFeature> active_features_;  // all active (real + dummy)
  std::vector<char> is_active_;          // O(1) lookup for reals

  // Basis (orthonormalized columns for VD bookkeeping)
  MatC basis_;
  std::vector<int> basis_indices_;
  inline int basis_size_() const noexcept { return static_cast<int>(basis_indices_.size()); }

  // Reusable buffers
  mutable Vec proj_coeffs_;
  mutable Vec ortho_buffer_;

  // Active columns and Cholesky factor of X_A^T X_A (unsigned)
  MatC X_active_;
  MatC chol_factor_;
  Vec  Xty_active_;                      // X_A^T y, built incrementally

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

  // ---------- VD machinery (shared logic with VD-LARS) ----------
  std::optional<Vec> orthonormalize_(const Vec& v) const;
  Vec  project_to_Vperp_(const Vec& z) const;
  void initialize_virtual_dummies_();
  void update_virtual_dummies_();
  void realize_dummy_(int vd_idx);

  // ---------- OMP-specific core steps ----------

  // Find the inactive candidate with largest |corr|.
  // Returns {kind, index, corr_value} or nullopt if nothing above eps.
  struct Candidate {
    enum class Pool : uint8_t { Real, VD, Realized };
    Pool   pool;
    int    index;
    double abs_corr;
  };
  std::optional<Candidate> find_best_candidate_() const;

  // Append a column to X_active_ / Cholesky factor (unsigned).
  void append_to_factor_(const Vec& x_col);

  // OLS refit: solve β_A = R^{-1} R^{-T} X_A^T y, update μ, r, correlations.
  void ols_refit_();

  // Disallow copying
  VD_OMP(const VD_OMP&) = delete;
  VD_OMP& operator=(const VD_OMP&) = delete;
};