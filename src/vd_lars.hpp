// vd_lars.hpp — Virtual Dummy LARS (equiangular direction, γ-step)
#pragma once
#include "vd_base.hpp"

class VD_LARS : public VD_Base {
public:
  using VD_Base::VD_Base;

  MatC run(int T = 1) override;

  Eigen::VectorXd corr_realized_view() const { return corr_realized_view_copy(); }

private:
  // LARS-specific state
  std::vector<int> pos_of_;
  std::unordered_set<int> actives_set_;
  Vec signs_, signs_dummy_;

  struct Direction {
    double A_active = 0.0;
    Vec w, u, a, a_vd, a_rd;
  };

  void   init_lars_();
  bool   lars_inited_ = false;

  double find_and_add_active_();
  void   update_factor_();
  Direction compute_direction_();
  double take_step_(double C, double A_active,
                    const Vec& w, const Vec& u,
                    const Vec& a, const Vec& a_vd, const Vec& a_rd);
};
