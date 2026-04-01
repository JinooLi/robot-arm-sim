#pragma once

#include "barrier.h"
#include "qp_solver.h"

#include <Eigen/Dense>
#include <tuple>
#include <utility>
#include <vector>

namespace cbf_core {

/// High-level orchestrator: barrier computation + safety-filter QP.
///
/// Python calls set_params() once (or when config changes), then
/// compute_safety_torque() every control step with pre-computed
/// Pinocchio data.
class CbfSolver {
public:
    explicit CbfSolver(int n_dof);

    void set_params(
        double gamma,
        double alpha_1,
        double alpha_2_star,
        double p1,
        double beta_1,
        double beta_2,
        const Eigen::VectorXd& qdot_max,
        const Eigen::VectorXd& tau_max,
        const Eigen::VectorXd& link_radii,
        const std::vector<std::pair<int, int>>& self_pairs);

    /// Main hot-path function.
    /// @return (tau, alpha_2, h_val)
    std::tuple<Eigen::VectorXd, double, double> compute_safety_torque(
        const Eigen::VectorXd& tau_nom,
        const Eigen::VectorXd& qdot,
        const Eigen::MatrixXd& M_inv,
        const Eigen::VectorXd& f2,
        // Link data at q
        const Eigen::MatrixXd& link_pos,
        const std::vector<Eigen::MatrixXd>& link_jac,
        // Link data at q + eps*qdot
        const Eigen::MatrixXd& link_pos_plus,
        const std::vector<Eigen::MatrixXd>& link_jac_plus,
        // Link data at q - eps*qdot
        const Eigen::MatrixXd& link_pos_minus,
        const std::vector<Eigen::MatrixXd>& link_jac_minus,
        // Obstacles
        const Eigen::MatrixXd& obs_positions,
        const Eigen::VectorXd& obs_radii);

private:
    int n_dof_;
    SafetyQpSolver qp_;

    // Cached parameters
    double gamma_ = 500.0;
    double alpha_1_ = 20.0;
    double alpha_2_star_ = 30.0;
    double p1_ = 1.0;
    double beta_1_ = 50.0;
    double beta_2_ = 50.0;
    Eigen::VectorXd qdot_max_;
    Eigen::VectorXd tau_max_;
    Eigen::VectorXd link_radii_;
    std::vector<std::pair<int, int>> self_pairs_;

    static constexpr double kFiniteDiffEps = 1e-7;
};

}  // namespace cbf_core
