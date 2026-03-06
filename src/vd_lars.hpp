// vd_lars.hpp
#pragma once
#include "vd_common.hpp"

class VD_LARS {
public:
  // Constructors (X,y are mapped views; no copies)
  VD_LARS(const double* Xptr, int n, int p,
          const double* yptr, int ny,
          int num_dummies,
          const VDOptions& o);

  VD_LARS(const Eigen::Ref<const MatC>& X_in,
          const Eigen::Ref<const Vec>&  y_in,
          int num_dummies,
          const VDOptions& o);

  // Top-level API
  MatC  run(int T = 1);                   // advance until at least T dummies realized (or no progress)
  int   num_dummies() const noexcept { return L_; }
  int   num_realized_dummies() const noexcept { return T_realized_; }
  int   n_features() const noexcept { return p_; }
  int   n_samples()  const noexcept { return n_; }
  int basis_size() const noexcept { return basis_size_(); }

  // Light, safe views/copies
  Eigen::VectorXd beta_view_copy() const { return beta_; }

  // Old names some code refers to
  Eigen::VectorXi active_indices() const { return actives_indices_(); }
  Eigen::VectorXd corr_realized_view() const { return corr_realized_view_copy(); }
  Eigen::VectorXd corr_view_copy() const { return corr_; }
  Eigen::VectorXd corr_realized_view_copy() const { return corr_realized_.head(T_realized_); }
  Eigen::VectorXi is_dummy_realized_view() const {
    Eigen::VectorXi out(L_);
    for (int d = 0; d < L_; ++d) out(d) = vd_is_realized_[d] ? 1 : 0;
    return out;
  }

  struct ActiveFeature {
    // Use 'enum class' for better scoping and type safety
    enum class Kind : uint8_t { Real, Dummy }; 
    Kind kind;
    int  index;
  };

  static constexpr int Y_SENTINEL     = -1;
  static constexpr int DUMMY_SENTINEL = -2;

  std::vector<ActiveFeature> active_features_copy() const { return active_features_; }

  Eigen::VectorXd beta_real() const {
    Eigen::VectorXd out(p_);
    if (beta_.size() >= p_) {
      out = beta_.head(p_);
    } else {
      out.setZero();
    }
    return out;
  }

  // Optional read-only references (avoid copying large arrays)
  const Vec&  normx_view()   const noexcept { return normx_; }
  const MatR& vd_proj_view() const noexcept { return vd_proj_; }
  const Vec&  vd_corr_view() const noexcept { return vd_corr_; }
  const Vec&  vd_stick_view()const noexcept { return vd_stick_; }

private:
  // ---------- Types ----------
  struct Direction {
    double A_active = 0.0;
    Vec w, u, a, a_vd, a_rd;
  };

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
  Vec normx_;                      // (optional) per-feature scale
  Vec beta_, beta_dummy_, mu_, residuals_, corr_;

  // Active sets & signs
  std::vector<int> actives_;
  Eigen::VectorXi actives_indices_() const;
  std::vector<ActiveFeature> active_features_;
  std::vector<char> is_active_;    // O(1) lookup
  std::vector<int>  pos_of_;       // index in actives_
  Vec  signs_, signs_dummy_;

  // Basis (orthonormalized packed columns), and active block
  MatC basis_;                     // n x (#basis)
  MatC X_active_;                  // n x (#active)
  std::vector<int> basis_indices_; // parallel to basis_ cols; -1 for y, -2 for realized dummy, >=0 for real j
  inline int basis_size_() const noexcept { return static_cast<int>(basis_indices_.size()); }

  // Reusable buffers
  mutable Vec proj_coeffs_;        // length <= min(n,p)
  mutable Vec ortho_buffer_;       // length n

  // Cholesky factor of X_active^T X_active
  MatC chol_factor_;

  // Realized dummies
  MatC X_realized_;                // n x T_max
  Vec  corr_realized_;             // size T_max
  int  T_realized_ = 0;

  // Virtual-dummy pool
  MatR vd_proj_;                   // rows = basis rows used, cols = L_
  Vec  vd_stick_, vd_corr_;
  std::vector<char> vd_is_realized_;
  std::vector<int>  vd_unrealized_idx_;

  // vd_proj capacity mgmt
  int vd_rows_filled_ = 0, vd_rows_cap_ = 0;

  // Bookkeeping
  std::unordered_set<int> actives_set_;
  int   step_ = 0;

  // ---------- Setup ----------
  void init_state_();
  void initialize_basis_();
  void ensure_vd_rows_capacity_(int need);

  // ---------- Core steps ----------
  std::optional<Vec> orthonormalize_(const Vec& v) const; 
  Vec project_to_Vperp_(const Vec& z) const;
  void   initialize_virtual_dummies_();
  static bool chol_append(MatC& R,
                          const Eigen::Ref<const Vec>& v,
                          double s,
                          double eps);
  void   update_virtual_dummies_();
  void   realize_dummy_(int vd_idx);
  double find_and_add_active_();
  void   update_factor_();
  Direction compute_direction_();
  double take_step_(double C, double A_active,
                    const Vec& w, const Vec& u,
                    const Vec& a, const Vec& a_vd, const Vec& a_rd);

  // ---------- RNG helpers ----------
  static double gamma_mt_(double m, std::mt19937_64& rng); // Marsaglia–Tsang Gamma(m,1)

  // ---------- Tiny utils ----------
  static inline double dabs_(double x) noexcept { return x < 0 ? -x : x; }

  // Disallow copying by default (instances keep internal views & state)
  VD_LARS(const VD_LARS&) = delete;
  VD_LARS& operator=(const VD_LARS&) = delete;
};