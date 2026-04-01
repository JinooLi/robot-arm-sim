#pragma once

#include <Eigen/Dense>
#include <memory>

// Forward-declare qpOASES types to avoid exposing the header publicly.
namespace qpOASES {
class SQProblem;
}

namespace cbf_core {

struct QpResult {
    Eigen::VectorXd tau;
    double alpha_2;
    bool success;
};

/// Thin wrapper around qpOASES for the safety-filter QP (Eq. 52).
///
/// Decision variable:  x = [τ (n_dof), α₂ (1)]
/// Objective:          min 0.5‖τ-τ_nom‖² + 0.5·p₁·(α₂-α₂*)²
/// Linear constraints: A·x ≥ b   (HOCBF + velocity CBFs)
/// Box bounds:         -τ_max ≤ τ ≤ τ_max,  α₂ ≥ 0
///
/// Uses SQProblem (sequential QP) to support hotstarting with
/// updated Hessian and constraint matrix.
class SafetyQpSolver {
public:
    explicit SafetyQpSolver(int n_dof);
    ~SafetyQpSolver();

    // Non-copyable
    SafetyQpSolver(const SafetyQpSolver&) = delete;
    SafetyQpSolver& operator=(const SafetyQpSolver&) = delete;

    QpResult solve(
        const Eigen::VectorXd& tau_nom,
        const Eigen::VectorXd& qdot,
        double h_val,
        const Eigen::VectorXd& grad_h,
        double hqq,
        const Eigen::MatrixXd& M_inv,
        const Eigen::VectorXd& f2,
        double alpha_1,
        double alpha_2_star,
        double p1,
        double beta_1,
        double beta_2,
        const Eigen::VectorXd& qdot_max,
        const Eigen::VectorXd& tau_max);

private:
    int n_dof_;
    int n_var_;   // n_dof + 1
    int n_con_;   // 1 + 2*n_dof

    // Pre-allocated buffers (row-major for qpOASES)
    using RowMatrixXd =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMatrixXd H_;
    Eigen::VectorXd g_;
    RowMatrixXd A_;
    Eigen::VectorXd lbA_, ubA_;
    Eigen::VectorXd lb_, ub_;

    std::unique_ptr<qpOASES::SQProblem> qp_;
    bool first_call_;
};

}  // namespace cbf_core
