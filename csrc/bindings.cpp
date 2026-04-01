#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cbf_core/barrier.h"
#include "cbf_core/cbf_solver.h"

namespace py = pybind11;
using namespace cbf_core;

PYBIND11_MODULE(_cbf_core, m) {
    m.doc() = "C++ acceleration module for robust CBF controller";

    // ---- Low-level barrier function (for testing / debugging) ----
    m.def(
        "compute_barrier",
        &compute_barrier,
        py::arg("link_positions"),
        py::arg("link_jacobians"),
        py::arg("link_radii"),
        py::arg("obs_positions"),
        py::arg("obs_radii"),
        py::arg("self_pairs"),
        py::arg("gamma"),
        py::arg("n_dof"),
        "Compute log-sum-exp barrier h(q) and gradient.");

    py::class_<BarrierResult>(m, "BarrierResult")
        .def_readonly("h_val", &BarrierResult::h_val)
        .def_readonly("grad_h", &BarrierResult::grad_h);

    // ---- High-level solver ----
    py::class_<CbfSolver>(m, "CbfSolver")
        .def(py::init<int>(), py::arg("n_dof"))
        .def(
            "set_params",
            &CbfSolver::set_params,
            py::arg("gamma"),
            py::arg("alpha_1"),
            py::arg("alpha_2_star"),
            py::arg("p1"),
            py::arg("beta_1"),
            py::arg("beta_2"),
            py::arg("qdot_max"),
            py::arg("tau_max"),
            py::arg("link_radii"),
            py::arg("self_pairs"))
        .def(
            "compute_safety_torque",
            &CbfSolver::compute_safety_torque,
            py::arg("tau_nom"),
            py::arg("qdot"),
            py::arg("M_inv"),
            py::arg("f2"),
            py::arg("link_pos"),
            py::arg("link_jac"),
            py::arg("link_pos_plus"),
            py::arg("link_jac_plus"),
            py::arg("link_pos_minus"),
            py::arg("link_jac_minus"),
            py::arg("obs_positions"),
            py::arg("obs_radii"));
}
