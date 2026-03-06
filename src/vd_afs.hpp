// vd_afs.hpp — Virtual Dummy Adaptive Forward Stepwise
// At ρ=1 reduces to VD-OMP.  At ρ→0 approaches LARS.
#pragma once
#include "vd_base.hpp"

class VD_AFS : public VD_Base {
public:
  using VD_Base::VD_Base;

  MatC run(int T = 1) override;
  double rho() const noexcept { return rho_; }

private:
  double rho_ = 1.0;
  Vec Xty_active_;
  Vec nu_active_;
  bool nu_stale_ = true;

  struct Candidate {
    enum class Pool : uint8_t { Real, VD, RealizedDummy };
    Pool pool; int index; double abs_corr; bool is_new;
  };
  std::optional<Candidate> find_best_candidate_() const;
  void append_to_factor_(const Vec& x_col);
  void ols_solve_();
  void afs_blend_();

  void init_afs_();
  bool afs_inited_ = false;
};
