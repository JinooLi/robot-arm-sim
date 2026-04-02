"""Robust whole-body safety-critical controller.

Based on: Xiong, Zhai & Xia, "Robust Whole-Body Safety-Critical Control for
Sampled-Data Robotic Manipulators via Control Barrier Functions",
IEEE Trans. Automation Science and Engineering, vol. 22, 2025.

Safety filter QP with:
  - Whole-body safety via log-sum-exp HOCBF  (Eq. 16, 48)
  - Joint velocity CBF constraints           (Eq. 35, 37)
Nominal controller: Lyapunov-based EE tracking converted to torque.

When the C++ extension (_cbf_core) is available, the barrier computation
and QP solve are accelerated via Eigen + qpOASES.  Otherwise the pure-Python
fallback (scipy SLSQP) is used transparently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..dynamics.pinocchio_model import PinocchioModel
from ..interfaces.controller import (
    ControllerInterface,
    ControlMode,
    ControlOutput,
)
from ..interfaces.simulator import RobotState

# Try to import C++ acceleration module
try:
    from .._cbf_core import CbfSolver as _CppCbfSolver

    _HAS_CPP = True
except ImportError:
    _CppCbfSolver = None
    _HAS_CPP = False

_N = PinocchioModel.ARM_DOF  # 7

# Finite-difference step for Hessian approximation
_FD_EPS = 1e-7


@dataclass
class SphereSpec:
    """Specification for a single collision sphere attached to a link."""

    link: int  # parent link index (0-6 for joints, 7 for EE)
    offset: np.ndarray  # 3D offset in the link's local frame
    radius: float

    @staticmethod
    def from_config(entry: dict) -> SphereSpec:
        return SphereSpec(
            link=entry["link"],
            offset=np.array(entry.get("offset", [0, 0, 0]), dtype=float),
            radius=float(entry["radius"]),
        )

    @staticmethod
    def from_radii(radii: list[float]) -> list[SphereSpec]:
        """Backward compat: one sphere per link at origin."""
        return [
            SphereSpec(link=i, offset=np.zeros(3), radius=r)
            for i, r in enumerate(radii)
        ]


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]×."""
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


class RobustCBFController(ControllerInterface):

    def __init__(self, dynamics: PinocchioModel) -> None:
        self._dyn = dynamics
        self._mode = ControlMode.TORQUE

        # ---- HOCBF (Sec. III-C) ----
        self._gamma: float = 500.0  # log-sum-exp smoothing  (Eq. 16)
        self._alpha_1: float = 20.0  # class-K coeff for ψ_1 (Eq. 39)
        self._alpha_2_star: float = 30.0  # desired α_2            (Eq. 52)
        self._p1: float = 1.0  # slack weight           (Eq. 52)

        # ---- Velocity CBF (Sec. III-B) ----
        self._beta_1: float = 50.0  # upper bound gain       (Eq. 21)
        self._beta_2: float = 50.0  # lower bound gain       (Eq. 21)

        # ---- Limits ----
        self._qdot_max = np.full(_N, 1.2)
        self._tau_max = np.array([50.0, 50, 50, 50, 8, 8, 8])

        # ---- Link spherical enclosures (Fig. 2) ----
        # Default: 7 joints + end-effector = 8 spheres at link origins
        self._spheres: list[SphereSpec] = SphereSpec.from_radii(
            [0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.06, 0.05]
        )
        self._sphere_radii = np.array([s.radius for s in self._spheres])

        # ---- Nominal controller ----
        self._kv: float = 20.0  # velocity tracking gain (computed torque)
        self._vel_scale: float = 1.0

        # ---- Self-collision (Eq. 15) ----
        self._use_self_col: bool = False
        self._self_pairs = self._build_self_pairs()

        self._prev_tau = np.zeros(_N)

        # C++ solver (created after setup if available)
        self._cpp: _CppCbfSolver | None = None

    def _build_self_pairs(self) -> list[tuple[int, int]]:
        """Non-adjacent sphere pairs for self-collision avoidance.

        Excluded pairs:
        - Spheres on the same parent link (rigidly attached, can't collide).
        - Spheres whose list indices differ by at most 1 (neighbours along
          the kinematic chain).
        """
        n = len(self._spheres)
        pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 2, n):
                if self._spheres[i].link != self._spheres[j].link:
                    pairs.append((i, j))
        return pairs

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, config: dict[str, Any]) -> None:
        ctrl = config["controller"]
        c = ctrl.get("robust_cbf", {})

        self._gamma = c.get("gamma", self._gamma)
        self._alpha_1 = c.get("alpha_1", self._alpha_1)
        self._alpha_2_star = c.get("alpha_2_star", self._alpha_2_star)
        self._p1 = c.get("p1", self._p1)
        self._beta_1 = c.get("beta_1", self._beta_1)
        self._beta_2 = c.get("beta_2", self._beta_2)

        if "q_dot_max" in c:
            self._qdot_max = np.array(c["q_dot_max"], dtype=float)
        if "tau_max" in c:
            self._tau_max = np.array(c["tau_max"], dtype=float)

        # Sphere configuration: prefer link_spheres, fall back to link_radii
        if "link_spheres" in c:
            self._spheres = [SphereSpec.from_config(s) for s in c["link_spheres"]]
        elif "link_radii" in c:
            self._spheres = SphereSpec.from_radii(c["link_radii"])
        self._sphere_radii = np.array([s.radius for s in self._spheres])
        self._self_pairs = self._build_self_pairs()

        self._kv = c.get("kv", self._kv)
        self._vel_scale = c.get("velocity_scale", self._vel_scale)
        self._use_self_col = c.get("self_collision", self._use_self_col)

        # Initialize C++ solver if available
        if _HAS_CPP:
            self._cpp = _CppCbfSolver(_N)
            self._cpp.set_params(
                self._gamma,
                self._alpha_1,
                self._alpha_2_star,
                self._p1,
                self._beta_1,
                self._beta_2,
                self._qdot_max,
                self._tau_max,
                self._sphere_radii,
                self._self_pairs if self._use_self_col else [],
            )
            print("[RobustCBFController] Using C++ backend (qpOASES)")
        else:
            print("[RobustCBFController] Using Python fallback (scipy SLSQP)")

        # Diagnose initial barrier values at the starting configuration
        q0 = np.array(config["robot"]["initial_joint_positions"], dtype=float)
        self._diagnose_barrier(q0, config.get("obstacles", []))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _diagnose_barrier(
        self,
        q: np.ndarray,
        obstacles: list[dict],
    ) -> None:
        """Print per-pair h_k at a given configuration to find penetrations."""
        sphere_pos, _ = self._compute_sphere_data(q)
        n = len(self._spheres)

        def _label(i: int) -> str:
            s = self._spheres[i]
            return f"S{i}(L{s.link} r={s.radius:.3f} off={s.offset.tolist()})"

        violations: list[str] = []

        # Obstacle pairs
        for si in range(n):
            for obs in obstacles:
                o_pos = np.array(obs["position"], dtype=float)
                o_r = float(obs["radius"])
                d = sphere_pos[si] - o_pos
                h_k = float(d @ d) - (self._sphere_radii[si] + o_r) ** 2
                if h_k < 0:
                    violations.append(
                        f"  OBS  {_label(si)} <-> obs@{obs['position']} r={o_r}  "
                        f"h_k={h_k:.6f}  dist={np.linalg.norm(d):.4f}  "
                        f"r_sum={self._sphere_radii[si] + o_r:.4f}"
                    )

        # Self-collision pairs
        if self._use_self_col:
            for i, j in self._self_pairs:
                d = sphere_pos[i] - sphere_pos[j]
                r_sum = self._sphere_radii[i] + self._sphere_radii[j]
                h_k = float(d @ d) - r_sum**2
                if h_k < 0:
                    violations.append(
                        f"  SELF {_label(i)} <-> {_label(j)}  "
                        f"h_k={h_k:.6f}  dist={np.linalg.norm(d):.4f}  "
                        f"r_sum={r_sum:.4f}"
                    )

        if violations:
            print(
                f"[RobustCBFController] WARNING: {len(violations)} negative h_k at initial pose:"
            )
            for v in violations:
                print(v)
        else:
            print("[RobustCBFController] All barrier values positive at initial pose.")

    # ------------------------------------------------------------------
    # Link data helper (shared by C++ and Python paths)
    # ------------------------------------------------------------------

    def _compute_sphere_data(
        self,
        q: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Compute world positions and 3×DOF Jacobians for all spheres.

        For spheres with a nonzero local-frame offset, the Jacobian is
        corrected:  J_sphere = J_v - [R·offset]× · J_ω
        """
        self._dyn.prepare(q)
        n_spheres = len(self._spheres)
        positions = np.empty((n_spheres, 3))
        jacobians: list[np.ndarray] = []

        for idx, sp in enumerate(self._spheres):
            if sp.link < _N:
                pos, R = self._dyn.get_link_placement(sp.link)
                J6 = self._dyn.get_link_jacobian_6d(sp.link)
            else:
                pos, R = self._dyn.get_ee_placement()
                J6 = self._dyn.get_ee_jacobian_6d()

            has_offset = np.any(sp.offset != 0)
            if has_offset:
                d = R @ sp.offset
                positions[idx] = pos + d
                # v_sphere = J_v·q̇ + ω × d = (J_v - [d]× · J_ω)·q̇
                jacobians.append(J6[:3] - _skew(d) @ J6[3:])
            else:
                positions[idx] = pos
                jacobians.append(J6[:3].copy())

        return positions, jacobians

    # ------------------------------------------------------------------
    # Whole-body safety constraint  h(q)  (Eq. 14-16)  [Python fallback]
    # ------------------------------------------------------------------

    def _h_and_grad(
        self,
        q: np.ndarray,
        obstacles: list[tuple[np.ndarray, float]],
    ) -> tuple[float, np.ndarray]:
        """Compute h(q) and ∇h(q) via log-sum-exp over all sphere pairs."""
        sphere_pos, sphere_jac = self._compute_sphere_data(q)
        n_spheres = len(self._spheres)

        h_vals: list[float] = []
        h_grads: list[np.ndarray] = []

        # Obstacle avoidance  h_i = ||X_i - O||² - (r_i + r_o)²  (Eq. 14)
        for si in range(n_spheres):
            for o_pos, o_r in obstacles:
                d = sphere_pos[si] - o_pos
                r_sum = self._sphere_radii[si] + o_r
                h_vals.append(float(d @ d) - r_sum**2)
                h_grads.append(2.0 * sphere_jac[si].T @ d)

        # Self-collision  h_{j,k}  (Eq. 15)
        if self._use_self_col:
            for i, j in self._self_pairs:
                d = sphere_pos[i] - sphere_pos[j]
                r_sum = self._sphere_radii[i] + self._sphere_radii[j]
                h_vals.append(float(d @ d) - r_sum**2)
                h_grads.append(2.0 * (sphere_jac[i] - sphere_jac[j]).T @ d)

        if not h_vals:
            return 1.0, np.zeros(_N)

        gamma = self._gamma
        h_arr = np.asarray(h_vals)

        # Numerically-stable log-sum-exp  (Eq. 16)
        a = -gamma * h_arr
        a_max = a.max()
        e = np.exp(a - a_max)
        s = e.sum()
        h_val = -(a_max + np.log(s)) / gamma

        w = e / s  # softmax weights
        grad_h = np.zeros(_N)
        for i in range(len(h_vals)):
            grad_h += w[i] * h_grads[i]

        return float(h_val), grad_h

    def _hess_qdot_qdot(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        obstacles: list[tuple[np.ndarray, float]],
    ) -> float:
        """q̇ᵀ ∇²h q̇  via central finite difference on ∇h."""
        eps = _FD_EPS
        _, g_plus = self._h_and_grad(q + eps * qdot, obstacles)
        _, g_minus = self._h_and_grad(q - eps * qdot, obstacles)
        return float(qdot @ (g_plus - g_minus)) / (2.0 * eps)

    # ------------------------------------------------------------------
    # Nominal controller  (Lyapunov tracking → computed torque)
    # ------------------------------------------------------------------

    def _nominal_torque(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        ee_pos: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        J = self._dyn.jacobian(q)[:3, :_N]
        J_pinv = np.linalg.pinv(J)
        # Lyapunov gradient descent: q̇_des = -scale · J⁺ (x_ee - x_target)
        qdot_des = -self._vel_scale * J_pinv @ (ee_pos - target)

        M = self._dyn.mass_matrix(q)
        nle = self._dyn.nonlinear_effects(q, qdot)
        return M @ (self._kv * (qdot_des - qdot)) + nle

    # ------------------------------------------------------------------
    # Safety-filter QP  (Eq. 52)  [Python fallback]
    # ------------------------------------------------------------------

    def _solve_safety_qp(
        self,
        tau_nom: np.ndarray,
        qdot: np.ndarray,
        h_val: float,
        grad_h: np.ndarray,
        hqq: float,
        M_inv: np.ndarray,
        f2: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """min ‖τ-τ_nom‖² + p₁(α₂-α₂*)²  s.t. CBF constraints."""
        from scipy.optimize import minimize as scipy_minimize

        n = _N
        a1 = self._alpha_1
        a2s = self._alpha_2_star
        p1 = self._p1
        b1 = self._beta_1
        b2 = self._beta_2
        qdm = self._qdot_max
        tm = self._tau_max

        # ---- Build A x ≥ b,  x = [τ(7), α₂(1)] ----
        rows: list[np.ndarray] = []
        rhs: list[float] = []

        # (1) HOCBF  ψ̃₂ ≥ 0  (Eq. 48, non-robust)
        gh_Minv = grad_h @ M_inv
        psi1 = float(grad_h @ qdot) + a1 * h_val
        c0 = hqq + float(grad_h @ f2) + a1 * float(grad_h @ qdot)

        row = np.zeros(n + 1)
        row[:n] = gh_Minv
        row[n] = psi1
        rows.append(row)
        rhs.append(-c0)

        # (2) Velocity upper CBF  ξ_{1,i} ≥ 0  (Eq. 35)
        for i in range(n):
            row = np.zeros(n + 1)
            row[:n] = -M_inv[i]
            rows.append(row)
            rhs.append(f2[i] - b1 * (qdm[i] - qdot[i]))

        # (3) Velocity lower CBF  ξ_{2,i} ≥ 0  (Eq. 37)
        for i in range(n):
            row = np.zeros(n + 1)
            row[:n] = M_inv[i]
            rows.append(row)
            rhs.append(-f2[i] - b2 * (qdm[i] + qdot[i]))

        A = np.array(rows)
        b_vec = np.array(rhs)

        def obj(x: np.ndarray) -> float:
            return 0.5 * np.sum((x[:n] - tau_nom) ** 2) + 0.5 * p1 * (x[n] - a2s) ** 2

        def jac(x: np.ndarray) -> np.ndarray:
            g = np.zeros(n + 1)
            g[:n] = x[:n] - tau_nom
            g[n] = p1 * (x[n] - a2s)
            return g

        bounds = [(-tm[i], tm[i]) for i in range(n)] + [(0.0, None)]
        x0 = np.concatenate([np.clip(tau_nom, -tm, tm), [a2s]])

        res = scipy_minimize(
            obj,
            x0,
            jac=jac,
            method="SLSQP",
            constraints={
                "type": "ineq",
                "fun": lambda x: A @ x - b_vec,
                "jac": lambda x: A,
            },
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-9},
        )

        if not res.success:
            return np.clip(tau_nom, -tm, tm), 0.0

        return res.x[:n].copy(), float(res.x[n])

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    def compute(
        self,
        state: RobotState,
        target: np.ndarray,
        obstacles: list[dict[str, Any]],
    ) -> ControlOutput:
        q = state.joint_positions
        qdot = state.joint_velocities
        ee = state.ee_position

        obs = [
            (np.array(o["position"], dtype=float), float(o["radius"]))
            for o in obstacles
        ]

        # No safety constraints → pure tracking
        if not obs and not self._use_self_col:
            return self._tracking_only(q, qdot, ee, target)

        # Dynamics quantities (always needed)
        M = self._dyn.mass_matrix(q)
        M_inv = np.linalg.inv(M)
        nle = self._dyn.nonlinear_effects(q, qdot)
        f2 = -M_inv @ nle

        # Nominal torque from Lyapunov tracking
        tau_nom = self._nominal_torque(q, qdot, ee, target)

        # ---- C++ fast path ----
        if self._cpp is not None:
            link_pos, link_jac = self._compute_sphere_data(q)
            link_pos_p, link_jac_p = self._compute_sphere_data(q + _FD_EPS * qdot)
            link_pos_m, link_jac_m = self._compute_sphere_data(q - _FD_EPS * qdot)

            # Pack obstacles into matrices
            n_obs = len(obs)
            obs_positions = np.array([o[0] for o in obs]).reshape(n_obs, 3)
            obs_radii = np.array([o[1] for o in obs])

            tau, alpha_2, h_val = self._cpp.compute_safety_torque(
                tau_nom,
                qdot,
                M_inv,
                f2,
                link_pos,
                link_jac,
                link_pos_p,
                link_jac_p,
                link_pos_m,
                link_jac_m,
                obs_positions,
                obs_radii,
            )
        else:
            # ---- Python fallback ----
            h_val, grad_h = self._h_and_grad(q, obs)
            hqq = self._hess_qdot_qdot(q, qdot, obs)
            tau, alpha_2 = self._solve_safety_qp(
                tau_nom,
                qdot,
                h_val,
                grad_h,
                hqq,
                M_inv,
                f2,
            )

        self._prev_tau = tau

        lyap = 0.5 * float((ee - target) @ (ee - target))
        return ControlOutput(
            command=tau,
            mode=self._mode,
            barrier_value=float(h_val),
            lyapunov_value=lyap,
            info={"alpha_2": alpha_2, "h_val": h_val},
        )

    def _tracking_only(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        ee: np.ndarray,
        target: np.ndarray,
    ) -> ControlOutput:
        tau = self._nominal_torque(q, qdot, ee, target)
        tau = np.clip(tau, -self._tau_max, self._tau_max)
        lyap = 0.5 * float((ee - target) @ (ee - target))
        return ControlOutput(
            command=tau,
            mode=self._mode,
            barrier_value=0.0,
            lyapunov_value=lyap,
        )

    def reset(self) -> None:
        self._prev_tau = np.zeros(_N)

    @property
    def spheres(self) -> list[SphereSpec]:
        """Sphere specifications (for visualization)."""
        return self._spheres

    @property
    def control_mode(self) -> ControlMode:
        return self._mode
