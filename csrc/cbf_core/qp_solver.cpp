#include "qp_solver.h"

#include <qpOASES.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace cbf_core {

SafetyQpSolver::SafetyQpSolver(int n_dof)
    : n_dof_(n_dof),
      n_var_(n_dof + 1),
      n_con_(1 + 2 * n_dof),
      H_(n_var_, n_var_),
      g_(n_var_),
      A_(n_con_, n_var_),
      lbA_(n_con_),
      ubA_(n_con_),
      lb_(n_var_),
      ub_(n_var_),
      first_call_(true) {

    // SQProblem supports hotstart with updated H and A
    qp_ = std::make_unique<qpOASES::SQProblem>(n_var_, n_con_);

    qpOASES::Options opts;
    opts.setToMPC();  // tuned for fast re-solves
    opts.printLevel = qpOASES::PL_NONE;
    qp_->setOptions(opts);

    H_.setZero();
    A_.setZero();
}

SafetyQpSolver::~SafetyQpSolver() = default;

QpResult SafetyQpSolver::solve(
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
    const Eigen::VectorXd& tau_max) {

    const int n = n_dof_;
    constexpr double INF = 1.0e20;

    // ---- Hessian H (diagonal) ----
    for (int i = 0; i < n; ++i) H_(i, i) = 1.0;
    H_(n, n) = p1;

    // ---- Linear term g ----
    g_.head(n) = -tau_nom;
    g_(n) = -p1 * alpha_2_star;

    // ---- Constraint matrix A (row-major) ----
    // Row 0: HOCBF  ψ̃₂ ≥ 0   (Eq. 48)
    Eigen::VectorXd gh_Minv = (grad_h.transpose() * M_inv).transpose();
    double psi1 = grad_h.dot(qdot) + alpha_1 * h_val;
    double c0 = hqq + grad_h.dot(f2) + alpha_1 * grad_h.dot(qdot);

    A_.row(0).head(n) = gh_Minv.transpose();
    A_(0, n) = psi1;
    lbA_(0) = -c0;
    ubA_(0) = INF;

    // Rows 1..n: Velocity upper CBF  (Eq. 35)
    for (int i = 0; i < n; ++i) {
        A_.row(1 + i).head(n) = -M_inv.row(i);
        A_(1 + i, n) = 0.0;
        lbA_(1 + i) = f2(i) - beta_1 * (qdot_max(i) - qdot(i));
        ubA_(1 + i) = INF;
    }

    // Rows n+1..2n: Velocity lower CBF  (Eq. 37)
    for (int i = 0; i < n; ++i) {
        A_.row(1 + n + i).head(n) = M_inv.row(i);
        A_(1 + n + i, n) = 0.0;
        lbA_(1 + n + i) = -f2(i) - beta_2 * (qdot_max(i) + qdot(i));
        ubA_(1 + n + i) = INF;
    }

    // ---- Box bounds ----
    for (int i = 0; i < n; ++i) {
        lb_(i) = -tau_max(i);
        ub_(i) = tau_max(i);
    }
    lb_(n) = 0.0;
    ub_(n) = INF;

    // ---- Solve ----
    qpOASES::int_t nWSR = 200;
    qpOASES::returnValue status;

    if (first_call_) {
        status = qp_->init(
            H_.data(), g_.data(), A_.data(),
            lb_.data(), ub_.data(), lbA_.data(), ubA_.data(),
            nWSR);
        if (status == qpOASES::SUCCESSFUL_RETURN) {
            first_call_ = false;
        }
    } else {
        // SQProblem::hotstart supports updated H, g, A, bounds
        status = qp_->hotstart(
            H_.data(), g_.data(), A_.data(),
            lb_.data(), ub_.data(), lbA_.data(), ubA_.data(),
            nWSR);

        // If hotstart fails, reset and try cold start
        if (status != qpOASES::SUCCESSFUL_RETURN) {
            qp_->reset();
            first_call_ = true;
            nWSR = 200;
            status = qp_->init(
                H_.data(), g_.data(), A_.data(),
                lb_.data(), ub_.data(), lbA_.data(), ubA_.data(),
                nWSR);
            if (status == qpOASES::SUCCESSFUL_RETURN) {
                first_call_ = false;
            }
        }
    }

    QpResult result;
    if (status == qpOASES::SUCCESSFUL_RETURN) {
        Eigen::VectorXd x(n_var_);
        qp_->getPrimalSolution(x.data());
        result.tau = x.head(n);
        result.alpha_2 = x(n);
        result.success = true;
    } else {
        // Fallback: clamp nominal torque
        result.tau = tau_nom.cwiseMax(-tau_max).cwiseMin(tau_max);
        result.alpha_2 = 0.0;
        result.success = false;
        first_call_ = true;
    }

    return result;
}

}  // namespace cbf_core
