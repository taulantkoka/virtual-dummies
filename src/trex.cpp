// trex.cpp
#include "trex.hpp"
#include <iostream>
#include <algorithm>

// ========================================================================
// Constructor
// ========================================================================
TRexSelector::TRexSelector(const TRexOptions& opts) : opt_(opts) {}

// ========================================================================
// Seeds
// ========================================================================
std::vector<unsigned long long> TRexSelector::make_seeds_(int K) const {
  std::mt19937_64 rng(opt_.seed);
  std::uniform_int_distribution<uint32_t> dist(0u, 2147483647u);
  std::vector<unsigned long long> seeds(K);
  for (int k = 0; k < K; ++k) seeds[k] = dist(rng);
  return seeds;
}

// ========================================================================
// VDOptions factory
// ========================================================================
VDOptions TRexSelector::make_vd_opts_(unsigned long long seed,
                                      int T_max, int n) const {
  VDOptions o;
  o.T_max       = T_max;
  o.max_vd_proj = std::min(n, (opt_.max_vd_proj > 0 ? opt_.max_vd_proj
                                                     : std::max(32, n)));
  o.eps         = opt_.eps;
  o.standardize = false;
  o.seed        = seed;
  o.dummy_law   = opt_.dummy_law;
  o.rho         = opt_.rho;
  o.mmap_fd         = opt_.mmap_fd;
  o.mmap_block_cols = opt_.mmap_block_cols;
  return o;
}

// ========================================================================
// Solver factory
// ========================================================================
std::unique_ptr<VD_Base> TRexSelector::make_solver_(
    const Eigen::Ref<const MatC>& X,
    const Eigen::Ref<const Vec>& y,
    int num_dummies, unsigned long long seed, int T_max) const
{
  VDOptions vd = make_vd_opts_(seed, T_max, (int)X.rows());
  switch (opt_.solver) {
    case SolverType::LARS:
      return std::make_unique<VD_LARS>(X, y, num_dummies, vd);
    case SolverType::OMP:
      return std::make_unique<VD_OMP>(X, y, num_dummies, vd);
    case SolverType::AFS:
      return std::make_unique<VD_AFS>(X, y, num_dummies, vd);
  }
  return std::make_unique<VD_LARS>(X, y, num_dummies, vd);
}

// ========================================================================
// Voting grid
// ========================================================================
Vec TRexSelector::make_V_(int K, double eps) const {
  std::vector<double> v;
  v.reserve(K + 1);
  for (int i = 0; i < K; ++i) {
    double val = 0.5 + double(i) / double(K);
    if (val < 1.0) v.push_back(val);
  }
  v.push_back(1.0 - eps);
  Vec V(v.size());
  for (int i = 0; i < (int)v.size(); ++i) V(i) = v[i];
  return V;
}

// ========================================================================
// Phi_prime
// ========================================================================
Vec TRexSelector::Phi_prime_fun_(
    int p, int T_stop, int num_dummies,
    const MatC& phi_T_mat, const Vec& Phi) const
{
  Vec av = phi_T_mat.colwise().sum();
  Vec delta_av = Vec::Zero(T_stop);
  for (int j = 0; j < p; ++j)
    if (Phi(j) > 0.5)
      delta_av.noalias() += phi_T_mat.row(j).transpose();

  Vec  delta_mod = delta_av;
  MatC phi_mod   = phi_T_mat;
  if (T_stop > 1) {
    delta_mod.segment(1, T_stop - 1) =
        delta_av.segment(1, T_stop - 1) - delta_av.segment(0, T_stop - 1);
    phi_mod.block(0, 1, p, T_stop - 1) =
        phi_T_mat.block(0, 1, p, T_stop - 1) - phi_T_mat.block(0, 0, p, T_stop - 1);
  }

  Vec phi_scale = Vec::Zero(T_stop);
  for (int t = 0; t < T_stop; ++t) {
    double denom = double(num_dummies) - (t + 1) + 1.0;
    if (delta_mod(t) > opt_.eps && denom > 0.0) {
      double numer = double(p) - av(t);
      phi_scale(t) = 1.0 - (numer / denom) / delta_mod(t);
    }
  }
  return phi_mod * phi_scale;
}

// ========================================================================
// FDP hat
// ========================================================================
Vec TRexSelector::fdp_hat_(const Vec& V, const Vec& Phi,
                           const Vec& Phi_prime) const {
  const int n_v = (int)V.size(), p = (int)Phi.size();
  Vec out = Vec::Constant(n_v, std::numeric_limits<double>::quiet_NaN());
  for (int i = 0; i < n_v; ++i) {
    double v = V(i);
    int R = 0; double num = 0.0;
    for (int j = 0; j < p; ++j)
      if (Phi(j) > v) { ++R; num += (1.0 - Phi_prime(j)); }
    out(i) = (R == 0) ? 0.0 : std::min(1.0, num / double(R));
  }
  return out;
}

// ========================================================================
// Variable selection: search (T, v) grid
// ========================================================================
TRexSelector::SelectResult TRexSelector::select_var_(
    int p, double tFDR, int T_stop,
    const Eigen::MatrixXd& FDP_hat_mat,
    const Eigen::MatrixXd& Phi_mat,
    const Vec& V) const
{
  const int n_v = (int)V.size();

  std::vector<int> T_cands;
  T_cands.reserve(T_stop);
  for (int t = 0; t < T_stop; ++t) {
    for (int j = 0; j < n_v; ++j)
      if (FDP_hat_mat(t, j) <= tFDR) { T_cands.push_back(t); break; }
  }

  SelectResult result;
  if (T_cands.empty()) {
    result.v_thresh = V(n_v - 1);
    result.selected_var = Eigen::VectorXi(0);
    return result;
  }

  int T_select = T_cands.back() + 1;
  int best_R = -1, best_t = -1, best_j = -1;

  for (int t = 0; t < T_select; ++t) {
    for (int j = 0; j < n_v; ++j) {
      if (FDP_hat_mat(t, j) > tFDR) continue;
      int cnt = 0;
      for (int i = 0; i < p; ++i)
        if (Phi_mat(t, i) > V(j)) ++cnt;
      if (cnt > best_R ||
          (cnt == best_R && (j > best_j || (j == best_j && t > best_t)))) {
        best_R = cnt; best_t = t; best_j = j;
      }
    }
  }

  result.v_thresh = V(best_j);
  std::vector<int> sel;
  for (int i = 0; i < p; ++i)
    if (Phi_mat(best_t, i) > result.v_thresh) sel.push_back(i);
  result.selected_var.resize((int)sel.size());
  for (int k = 0; k < (int)sel.size(); ++k) result.selected_var(k) = sel[k];
  return result;
}

// ========================================================================
// Posthoc mode: one parallel region, all K solvers run T_max independently
// ========================================================================
TRexResult TRexSelector::run_posthoc_(
    const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
    int Tmax, int num_dummies, int n_threads, const Vec& V)
{
  const int p = (int)X.cols();
  const int n_v = (int)V.size();

  auto seeds = make_seeds_(opt_.K);

  std::vector<std::unique_ptr<VD_Base>> solvers;
  solvers.reserve(opt_.K);
  for (int k = 0; k < opt_.K; ++k)
    solvers.push_back(make_solver_(X, y, num_dummies, seeds[k], Tmax));

  std::vector<MatC> per_solver_phi(opt_.K);
  for (int k = 0; k < opt_.K; ++k)
    per_solver_phi[k] = MatC::Zero(p, Tmax);

  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
  #endif
  for (int k = 0; k < opt_.K; ++k) {
    for (int t = 1; t <= Tmax; ++t) {
      solvers[k]->run(t);
      for (const auto& af : solvers[k]->active_features_copy()) {
        if (af.kind == ActiveFeature::Kind::Real) {
          int j = af.index;
          if (j >= 0 && j < p) per_solver_phi[k](j, t - 1) = 1.0;
        }
      }
    }
  }

  solvers.clear();

  MatC phi_T_mat = MatC::Zero(p, Tmax);
  for (int k = 0; k < opt_.K; ++k) phi_T_mat += per_solver_phi[k];
  phi_T_mat.array() /= double(opt_.K);
  per_solver_phi.clear();

  Eigen::MatrixXd FDP_hat_mat(Tmax, n_v);
  Eigen::MatrixXd Phi_mat(Tmax, p);
  Vec Phi_prime;

  for (int t = 1; t <= Tmax; ++t) {
    Vec Phi_t = phi_T_mat.col(t - 1);
    Phi_mat.row(t - 1) = Phi_t.transpose();
    Vec pp = Phi_prime_fun_(p, t, num_dummies, phi_T_mat.leftCols(t), Phi_t);
    Vec fh = fdp_hat_(V, Phi_t, pp);
    FDP_hat_mat.row(t - 1) = fh.transpose();
    if (t == Tmax) Phi_prime = std::move(pp);
  }

  SelectResult sel = select_var_(p, opt_.tFDR, Tmax, FDP_hat_mat, Phi_mat, V);

  TRexResult out;
  out.selected_var  = sel.selected_var;
  out.v_thresh      = sel.v_thresh;
  out.T_stop        = Tmax;
  out.num_dummies   = num_dummies;
  out.L_calibrated  = num_dummies;
  out.V             = V;
  out.FDP_hat_mat   = std::move(FDP_hat_mat);
  out.Phi_mat       = std::move(Phi_mat);
  out.Phi_prime     = std::move(Phi_prime);
  out.K             = opt_.K;
  return out;
}

// ========================================================================
// Early-stop mode: strided barriers
// ========================================================================
TRexResult TRexSelector::run_early_stop_(
    const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
    int Tmax, int num_dummies, int n_threads, const Vec& V)
{
  const int p = (int)X.cols();
  const int n_v = (int)V.size();
  const int SW = std::max(1, opt_.stride_width);

  auto seeds = make_seeds_(opt_.K);

  std::vector<std::unique_ptr<VD_Base>> solvers;
  solvers.reserve(opt_.K);
  for (int k = 0; k < opt_.K; ++k)
    solvers.push_back(make_solver_(X, y, num_dummies, seeds[k], Tmax));

  MatC phi_T_mat = MatC::Zero(p, Tmax);
  int T_stop = 0;

  while (T_stop < Tmax) {
    const int T_next = std::min(T_stop + SW, Tmax);
    const int steps  = T_next - T_stop;

    std::vector<MatC> local_masks(opt_.K);
    for (int k = 0; k < opt_.K; ++k)
      local_masks[k] = MatC::Zero(p, steps);

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
    #endif
    for (int k = 0; k < opt_.K; ++k) {
      for (int s = 0; s < steps; ++s) {
        solvers[k]->run(T_stop + s + 1);
        for (const auto& af : solvers[k]->active_features_copy()) {
          if (af.kind == ActiveFeature::Kind::Real) {
            int j = af.index;
            if (j >= 0 && j < p) local_masks[k](j, s) = 1.0;
          }
        }
      }
    }

    for (int s = 0; s < steps; ++s) {
      Vec Phi_t = Vec::Zero(p);
      for (int k = 0; k < opt_.K; ++k) Phi_t += local_masks[k].col(s);
      Phi_t.array() /= double(opt_.K);
      phi_T_mat.col(T_stop + s) = Phi_t;
    }

    T_stop = T_next;

    Vec Phi_last = phi_T_mat.col(T_stop - 1);
    Vec pp = Phi_prime_fun_(p, T_stop, num_dummies,
                            phi_T_mat.leftCols(T_stop), Phi_last);
    Vec fh = fdp_hat_(V, Phi_last, pp);

    if (opt_.verbose)
      std::cout << "[TRex] T=" << T_stop
                << " FDP_hat(max_v)=" << fh(n_v - 1) << "\n";

    if (fh(n_v - 1) > opt_.tFDR) break;
  }

  solvers.clear();

  const int T_rows = T_stop;
  Eigen::MatrixXd FDP_hat_mat(T_rows, n_v);
  Eigen::MatrixXd Phi_mat(T_rows, p);
  Vec Phi_prime;

  for (int t = 1; t <= T_rows; ++t) {
    Vec Phi_t = phi_T_mat.col(t - 1);
    Phi_mat.row(t - 1) = Phi_t.transpose();
    Vec pp = Phi_prime_fun_(p, t, num_dummies, phi_T_mat.leftCols(t), Phi_t);
    Vec fh = fdp_hat_(V, Phi_t, pp);
    FDP_hat_mat.row(t - 1) = fh.transpose();
    if (t == T_rows) Phi_prime = std::move(pp);
  }

  SelectResult sel = select_var_(p, opt_.tFDR, T_rows, FDP_hat_mat, Phi_mat, V);

  TRexResult out;
  out.selected_var  = sel.selected_var;
  out.v_thresh      = sel.v_thresh;
  out.T_stop        = T_rows;
  out.num_dummies   = num_dummies;
  out.L_calibrated  = num_dummies;
  out.V             = V;
  out.FDP_hat_mat   = std::move(FDP_hat_mat);
  out.Phi_mat       = std::move(Phi_mat);
  out.Phi_prime     = std::move(Phi_prime);
  out.K             = opt_.K;
  return out;
}

// ========================================================================
// L calibration: scan L = p, 2p, 3p, ... at T=1, v=0.75
// ========================================================================
int TRexSelector::calibrate_L_(
    const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
    int p, int n_threads)
{
  const double v_calib = 0.75;
  const int n = (int)X.rows();

  for (int L_mult = 1; L_mult <= opt_.max_L_factor; ++L_mult) {
    const int L = L_mult * p;

    if (opt_.verbose)
      std::cout << "[TRex] calibrate_L: trying L=" << L
                << " (" << L_mult << "p)\n";

    auto seeds = make_seeds_(opt_.K);

    // Run K experiments at T=1
    std::vector<std::unique_ptr<VD_Base>> solvers(opt_.K);
    for (int k = 0; k < opt_.K; ++k)
      solvers[k] = make_solver_(X, y, L, seeds[k], /*T_max=*/1);

    MatC phi_sum = MatC::Zero(p, 1);

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
    #endif
    for (int k = 0; k < opt_.K; ++k) {
      solvers[k]->run(1);
      Vec mask = Vec::Zero(p);
      for (const auto& af : solvers[k]->active_features_copy()) {
        if (af.kind == ActiveFeature::Kind::Real) {
          int j = af.index;
          if (j >= 0 && j < p) mask(j) = 1.0;
        }
      }
      #ifdef _OPENMP
      #pragma omp critical
      #endif
      phi_sum.col(0) += mask;
    }

    solvers.clear();
    phi_sum.array() /= double(opt_.K);
    Vec Phi = phi_sum.col(0);

    Vec pp = Phi_prime_fun_(p, 1, L, phi_sum, Phi);

    int R = 0; double num = 0.0;
    for (int j = 0; j < p; ++j)
      if (Phi(j) > v_calib) { ++R; num += (1.0 - pp(j)); }
    double fdp = (R == 0) ? 0.0 : std::min(1.0, num / double(R));

    if (opt_.verbose)
      std::cout << "[TRex] calibrate_L: L=" << L
                << " FDP_hat(v=0.75)=" << fdp << "\n";

    if (fdp <= opt_.tFDR) {
      if (opt_.verbose)
        std::cout << "[TRex] calibrate_L: accepted L=" << L << "\n";
      return L;
    }
  }

  int max_L = opt_.max_L_factor * p;
  if (opt_.verbose)
    std::cout << "[TRex] calibrate_L: ceiling, using L=" << max_L << "\n";
  return max_L;
}

// ========================================================================
// Main entry point
// ========================================================================
TRexResult TRexSelector::run(
    const Eigen::Ref<const MatC>& X,
    const Eigen::Ref<const Vec>& y)
{
  const int n = (int)X.rows(), p = (int)X.cols();

  // Thread count
  int n_threads = opt_.K;
  #ifdef _OPENMP
  if (opt_.n_threads > 0)
    n_threads = opt_.n_threads;
  else
    n_threads = std::min(opt_.K, omp_get_max_threads());
  #else
  n_threads = 1;
  #endif

  // Voting grid
  Vec V = make_V_(opt_.K, opt_.eps);

  if (opt_.verbose) {
    const char* solver_names[] = {"LARS", "OMP", "AFS"};
    const char* calib_names[]  = {"FixedTL", "CalibrateT", "CalibrateL", "CalibrateBoth"};
    std::cout << "[TRex] p=" << p << " n=" << n
              << " K=" << opt_.K
              << " solver=" << solver_names[(int)opt_.solver]
              << " calib=" << calib_names[(int)opt_.calib]
              << " threads=" << n_threads
              << " mode=" << (opt_.posthoc_mode ? "posthoc" : "early-stop")
              << "\n";
  }

  // ---- Determine L ----
  int num_dummies;
  switch (opt_.calib) {
    case CalibMode::FixedTL:
    case CalibMode::CalibrateT:
      num_dummies = opt_.L_factor * p;
      break;
    case CalibMode::CalibrateL:
    case CalibMode::CalibrateBoth:
      num_dummies = calibrate_L_(X, y, p, n_threads);
      break;
  }

  // ---- Determine T_max ----
  int Tmax;
  if (opt_.T_stop > 0) {
    Tmax = opt_.T_stop;
  } else {
    Tmax = std::min(num_dummies, (int)std::ceil(n / 2.0));
  }

  if (opt_.verbose)
    std::cout << "[TRex] using L=" << num_dummies
              << " T_max=" << Tmax << "\n";

  // ---- Dispatch ----
  TRexResult out;

  switch (opt_.calib) {
    case CalibMode::FixedTL:
    case CalibMode::CalibrateL:
      // Fixed T: always posthoc (no early stop needed)
      out = run_posthoc_(X, y, Tmax, num_dummies, n_threads, V);
      break;

    case CalibMode::CalibrateT:
    case CalibMode::CalibrateBoth:
      // Search over T: use posthoc or early-stop per user setting
      if (opt_.posthoc_mode)
        out = run_posthoc_(X, y, Tmax, num_dummies, n_threads, V);
      else
        out = run_early_stop_(X, y, Tmax, num_dummies, n_threads, V);
      break;
  }

  if (opt_.verbose)
    std::cout << "[TRex] selected " << out.selected_var.size()
              << " variables at v=" << out.v_thresh
              << " T=" << out.T_stop
              << " L=" << out.num_dummies << "\n";

  return out;
}
