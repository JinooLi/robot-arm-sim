#include "barrier.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cbf_core {

BarrierResult compute_barrier(
    const Eigen::MatrixXd& link_positions,
    const std::vector<Eigen::MatrixXd>& link_jacobians,
    const Eigen::VectorXd& link_radii,
    const Eigen::MatrixXd& obs_positions,
    const Eigen::VectorXd& obs_radii,
    const std::vector<std::pair<int, int>>& self_pairs,
    double gamma,
    int n_dof) {

    const int n_links = static_cast<int>(link_radii.size());
    const int n_obs = static_cast<int>(obs_radii.size());
    const int n_pairs = n_links * n_obs + static_cast<int>(self_pairs.size());

    if (n_pairs == 0) {
        return {1.0, Eigen::VectorXd::Zero(n_dof)};
    }

    // Pre-allocate h values and gradients
    Eigen::VectorXd h_vals(n_pairs);
    std::vector<Eigen::VectorXd> h_grads(n_pairs, Eigen::VectorXd(n_dof));

    int idx = 0;

    // --- Obstacle avoidance: h_i = ||X_i - O||^2 - (r_i + r_o)^2  (Eq. 14) ---
    for (int pi = 0; pi < n_links; ++pi) {
        const Eigen::Vector3d& p = link_positions.row(pi);
        const Eigen::MatrixXd& J = link_jacobians[pi];  // (3 x n_dof)

        for (int oi = 0; oi < n_obs; ++oi) {
            const Eigen::Vector3d d = p - obs_positions.row(oi).transpose();
            const double r_sum = link_radii[pi] + obs_radii[oi];
            h_vals[idx] = d.squaredNorm() - r_sum * r_sum;
            h_grads[idx] = 2.0 * J.transpose() * d;
            ++idx;
        }
    }

    // --- Self-collision: h_{j,k} (Eq. 15) ---
    for (const auto& [i, j] : self_pairs) {
        const Eigen::Vector3d d =
            link_positions.row(i).transpose() - link_positions.row(j).transpose();
        const double r_sum = link_radii[i] + link_radii[j];
        h_vals[idx] = d.squaredNorm() - r_sum * r_sum;
        h_grads[idx] =
            2.0 * (link_jacobians[i] - link_jacobians[j]).transpose() * d;
        ++idx;
    }

    // --- Numerically-stable log-sum-exp (Eq. 16) ---
    // h(q) = -(1/γ) ln Σ exp(-γ h_k)
    Eigen::VectorXd a = -gamma * h_vals;
    const double a_max = a.maxCoeff();
    Eigen::VectorXd e = (a.array() - a_max).exp();
    const double s = e.sum();
    const double h_val = -(a_max + std::log(s)) / gamma;

    // Softmax weights
    e /= s;

    // Weighted gradient
    Eigen::VectorXd grad_h = Eigen::VectorXd::Zero(n_dof);
    for (int k = 0; k < n_pairs; ++k) {
        grad_h += e[k] * h_grads[k];
    }

    return {h_val, grad_h};
}

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
    int n_dof) {

    // h(q) and ∇h(q) at the current configuration
    auto [h_val, grad_h] = compute_barrier(
        link_pos, link_jac, link_radii, obs_positions, obs_radii,
        self_pairs, gamma, n_dof);

    // ∇h at q + ε·q̇
    auto [_, grad_plus] = compute_barrier(
        link_pos_plus, link_jac_plus, link_radii, obs_positions, obs_radii,
        self_pairs, gamma, n_dof);

    // ∇h at q - ε·q̇
    auto [__, grad_minus] = compute_barrier(
        link_pos_minus, link_jac_minus, link_radii, obs_positions, obs_radii,
        self_pairs, gamma, n_dof);

    // q̇ᵀ ∇²h q̇ ≈ q̇ᵀ (∇h(q+ε·q̇) - ∇h(q-ε·q̇)) / (2ε)
    const double hqq = qdot.dot(grad_plus - grad_minus) / (2.0 * eps);

    return {h_val, grad_h, hqq};
}

}  // namespace cbf_core
