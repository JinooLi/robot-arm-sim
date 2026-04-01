#pragma once

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace cbf_core {

struct BarrierResult {
    double h_val;
    Eigen::VectorXd grad_h;
};

/// Compute log-sum-exp smooth-min barrier h(q) and its gradient (Eq. 14-16).
///
/// @param link_positions   (n_links x 3) matrix – precomputed link positions
/// @param link_jacobians   vector of (3 x n_dof) matrices – one per link
/// @param link_radii       (n_links,) vector of spherical enclosure radii
/// @param obs_positions    (n_obs x 3) matrix of obstacle centres
/// @param obs_radii        (n_obs,) vector of obstacle radii
/// @param self_pairs       non-adjacent link index pairs for self-collision
/// @param gamma            log-sum-exp smoothing parameter
/// @param n_dof            number of actuated joints
BarrierResult compute_barrier(
    const Eigen::MatrixXd& link_positions,
    const std::vector<Eigen::MatrixXd>& link_jacobians,
    const Eigen::VectorXd& link_radii,
    const Eigen::MatrixXd& obs_positions,
    const Eigen::VectorXd& obs_radii,
    const std::vector<std::pair<int, int>>& self_pairs,
    double gamma,
    int n_dof);

/// Combined barrier quantities needed for HOCBF constraint.
struct CbfQuantities {
    double h_val;
    Eigen::VectorXd grad_h;
    double hqq;  // q̇ᵀ ∇²h q̇  (via finite difference)
};

/// Compute h(q), ∇h(q), and q̇ᵀ∇²hq̇ in one call.
/// Requires precomputed link data at q, q+ε·q̇, and q-ε·q̇.
CbfQuantities compute_cbf_quantities(
    const Eigen::MatrixXd& link_pos,
    const std::vector<Eigen::MatrixXd>& link_jac,
    const Eigen::MatrixXd& link_pos_plus,
    const std::vector<Eigen::MatrixXd>& link_jac_plus,
    const Eigen::MatrixXd& link_pos_minus,
    const std::vector<Eigen::MatrixXd>& link_jac_minus,
    const Eigen::VectorXd& qdot,
    const Eigen::VectorXd& link_radii,
    const Eigen::MatrixXd& obs_positions,
    const Eigen::VectorXd& obs_radii,
    const std::vector<std::pair<int, int>>& self_pairs,
    double gamma,
    double eps,
    int n_dof);

}  // namespace cbf_core
