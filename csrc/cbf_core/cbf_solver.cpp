#include "cbf_solver.h"

namespace cbf_core {

CbfSolver::CbfSolver(int n_dof)
    : n_dof_(n_dof),
      qp_(n_dof),
      qdot_max_(Eigen::VectorXd::Constant(n_dof, 1.2)),
      tau_max_(Eigen::VectorXd::Constant(n_dof, 50.0)),
      link_radii_() {}

void CbfSolver::set_params(
    double gamma,
    double alpha_1,
    double alpha_2_star,
    double p1,
    double beta_1,
    double beta_2,
    const Eigen::VectorXd& qdot_max,
    const Eigen::VectorXd& tau_max,
    const Eigen::VectorXd& link_radii,
    const std::vector<std::pair<int, int>>& self_pairs) {

    gamma_ = gamma;
    alpha_1_ = alpha_1;
    alpha_2_star_ = alpha_2_star;
    p1_ = p1;
    beta_1_ = beta_1;
    beta_2_ = beta_2;
    qdot_max_ = qdot_max;
    tau_max_ = tau_max;
    link_radii_ = link_radii;
    self_pairs_ = self_pairs;
}

std::tuple<Eigen::VectorXd, double, double> CbfSolver::compute_safety_torque(
    const Eigen::VectorXd& tau_nom,
    const Eigen::VectorXd& qdot,
    const Eigen::MatrixXd& M_inv,
    const Eigen::VectorXd& f2,
    const Eigen::MatrixXd& link_pos,
    const std::vector<Eigen::MatrixXd>& link_jac,
    const Eigen::MatrixXd& link_pos_plus,
    const std::vector<Eigen::MatrixXd>& link_jac_plus,
    const Eigen::MatrixXd& link_pos_minus,
    const std::vector<Eigen::MatrixXd>& link_jac_minus,
    const Eigen::MatrixXd& obs_positions,
    const Eigen::VectorXd& obs_radii) {

    // Compute h(q), ∇h(q), q̇ᵀ∇²hq̇
    auto cbf = compute_cbf_quantities(
        link_pos, link_jac,
        link_pos_plus, link_jac_plus,
        link_pos_minus, link_jac_minus,
        qdot,
        link_radii_, obs_positions, obs_radii,
        self_pairs_, gamma_, kFiniteDiffEps, n_dof_);

    // Solve safety-filter QP
    auto qp_result = qp_.solve(
        tau_nom, qdot,
        cbf.h_val, cbf.grad_h, cbf.hqq,
        M_inv, f2,
        alpha_1_, alpha_2_star_, p1_,
        beta_1_, beta_2_,
        qdot_max_, tau_max_);

    return {qp_result.tau, qp_result.alpha_2, cbf.h_val};
}

}  // namespace cbf_core
