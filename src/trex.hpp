// trex.hpp — T-Rex selector with solver dispatch and calibration modes.
//
// Solver dispatch: SolverType selects LARS / OMP / AFS via VD_Base.
//
// Calibration modes:
//   FixedTL:       user-specified T and L, calibrate v only.
//   CalibrateT:    fix L = L_factor * p, search (T, v) grid with early stopping.
//   CalibrateL:    fix T, scan L = p, 2p, ... until FDP_hat(T=1, v=0.75) < α.
//   CalibrateBoth: calibrate L first, then search (T, v).
//
// Parallelism: OpenMP (falls back to sequential when not available).
#pragma once
#include "vd_base.hpp"
#include "vd_lars.hpp"
#include "vd_omp.hpp"
#include "vd_afs.hpp"
#include <memory>

#ifdef _OPENMP
  #include <omp.h>
#endif

// ---------- Enums ----------
enum class SolverType : uint8_t { LARS = 0, OMP = 1, AFS = 2 };
enum class CalibMode  : uint8_t { FixedTL = 0, CalibrateT = 1, CalibrateL = 2, CalibrateBoth = 3 };

// ---------- Options ----------
struct TRexOptions {
  double tFDR         = 0.2;
  int    K            = 20;        // number of random experiments
  int    L_factor     = 10;        // L = L_factor * p  (when L is fixed)
  int    T_stop       = -1;        // fixed T; -1 = auto (min(L, n/2))
  int    max_L_factor = 50;        // ceiling for L calibration

  int    stride_width  = 1;        // steps between early-stop checks
  bool   posthoc_mode  = false;    // posthoc (no early stop) vs strided early-stop
  int    max_vd_proj   = 100;
  double eps           = 1e-12;
  bool   verbose       = true;
  unsigned long long seed = 0ULL;

  SolverType solver    = SolverType::LARS;
  CalibMode  calib     = CalibMode::CalibrateBoth;
  VDDummyLaw dummy_law = VDDummyLaw::Spherical;
  double     rho       = 1.0;      // AFS shrinkage

  int mmap_fd         = -1;
  int mmap_block_cols = 0;
  int n_threads       = 0;         // 0 = auto
};

// ---------- Result ----------
struct TRexResult {
  Eigen::VectorXi selected_var;
  double          v_thresh      = 0.0;
  int             T_stop        = 0;
  int             num_dummies   = 0;
  int             L_calibrated  = 0;
  Eigen::VectorXd V;
  Eigen::MatrixXd FDP_hat_mat;    // (T_stop × |V|)
  Eigen::MatrixXd Phi_mat;        // (T_stop × p)
  Eigen::VectorXd Phi_prime;
  int             K = 0;
};

// ---------- Selector ----------
class TRexSelector {
public:
  explicit TRexSelector(const TRexOptions& opts);

  TRexResult run(const Eigen::Ref<const MatC>& X,
                 const Eigen::Ref<const Vec>&  y);

private:
  TRexOptions opt_;

  // ---- Solver factory ----
  std::unique_ptr<VD_Base> make_solver_(
      const Eigen::Ref<const MatC>& X,
      const Eigen::Ref<const Vec>& y,
      int num_dummies, unsigned long long seed, int T_max) const;

  VDOptions make_vd_opts_(unsigned long long seed, int T_max, int n) const;

  // ---- Execution paths ----
  TRexResult run_posthoc_(
      const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
      int Tmax, int num_dummies, int n_threads, const Vec& V);

  TRexResult run_early_stop_(
      const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
      int Tmax, int num_dummies, int n_threads, const Vec& V);

  // ---- L calibration ----
  int calibrate_L_(
      const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
      int p, int n_threads);

  // ---- Statistical helpers ----
  Vec make_V_(int K, double eps) const;

  Vec Phi_prime_fun_(int p, int T_stop, int num_dummies,
                     const MatC& phi_T_mat, const Vec& Phi) const;

  Vec fdp_hat_(const Vec& V, const Vec& Phi, const Vec& Phi_prime) const;

  struct SelectResult {
    Eigen::VectorXi selected_var;
    double v_thresh;
  };

  SelectResult select_var_(int p, double tFDR, int T_stop,
                           const Eigen::MatrixXd& FDP_hat_mat,
                           const Eigen::MatrixXd& Phi_mat,
                           const Vec& V) const;

  // ---- Seeds ----
  std::vector<unsigned long long> make_seeds_(int K) const;
};
