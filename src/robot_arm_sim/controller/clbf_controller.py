from __future__ import annotations

from typing import Any

import numpy as np

from ..dynamics.pinocchio_model import PinocchioModel
from ..interfaces.controller import (
    ControllerInterface,
    ControlMode,
    ControlOutput,
)
from ..interfaces.simulator import RobotState


class CLBFController(ControllerInterface):
    """CLBF (Control Lyapunov Barrier Function) controller.

    Combines a Lyapunov-based end-effector tracking controller with a
    barrier-function-based safety controller that avoids spherical obstacles.
    Uses softmax weighting across all arm links and the end-effector to
    smoothly blend safety corrections.
    """

    def __init__(self, dynamics: PinocchioModel) -> None:
        self._dynamics = dynamics
        self._mode = ControlMode.VELOCITY
        self._kp = np.ones(7) * 10.0

        # CLBF parameters
        self._k: float = 10.0
        self._s: float = 1.0
        self._barrier_radius_margin: float = 0.05
        self._alpha: float = 1.0
        self._velocity_scale: float = 10.0

    def setup(self, config: dict[str, Any]) -> None:
        ctrl_cfg = config["controller"]
        self._kp = np.array(ctrl_cfg["gains"]["kp"], dtype=float)
        self._mode = (
            ControlMode.VELOCITY
        )  # 이건 속도 제어만 한다. 토크 제어는 지원하지 않음.

        clbf_cfg = ctrl_cfg.get("clbf", {})
        self._k = clbf_cfg.get("k", 10.0)
        self._s = clbf_cfg.get("s", 50.0)
        self._barrier_radius_margin = clbf_cfg.get("barrier_radius_margin", 0.05)
        self._alpha = clbf_cfg.get("alpha", 1.0)

    # ---- Barrier / Lyapunov primitives ----

    def _sigmoid(self, s: float) -> float:
        return 1.0 / (1.0 + np.exp(-s))

    def _barrier_for_obstacle(
        self, x: np.ndarray, obs_center: np.ndarray, obs_radius: float
    ) -> float:
        """Compute barrier value B(x) for a single obstacle."""
        k = self._k
        s = self._s
        diff = x - obs_center
        circ = -0.5 * (diff @ diff - (obs_radius + self._barrier_radius_margin) ** 2)
        return self._sigmoid(circ * k * s) * k

    def _barrier_grad_for_obstacle(
        self, x: np.ndarray, obs_center: np.ndarray, obs_radius: float
    ) -> np.ndarray:
        """Compute dB/dx for a single obstacle."""
        k = self._k
        s = self._s
        B = self._barrier_for_obstacle(x, obs_center, obs_radius)
        dCirc_dx = -(x - obs_center)
        return s * B * (k - B) * dCirc_dx

    def _lyapunov(self, ee_pos: np.ndarray, target: np.ndarray) -> float:
        diff = ee_pos - target
        return 0.5 * float(diff @ diff)

    def _lyapunov_grad(self, ee_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        return ee_pos - target

    # ---- Main compute ----

    def compute(
        self,
        state: RobotState,
        target: np.ndarray,
        obstacles: list[dict[str, Any]],
    ) -> ControlOutput:
        q = state.joint_positions
        n_joints = PinocchioModel.ARM_DOF
        alpha = self._alpha

        if not obstacles:
            return self._tracking_only(state, target)

        # Parse obstacle data
        obs_list = [
            (np.array(o["position"], dtype=float), float(o["radius"]))
            for o in obstacles
        ]

        # Collect barrier values and gradients for all links x all obstacles
        # Points to check: joint links 0..6 + end-effector
        link_positions: list[np.ndarray] = []
        link_jacobians: list[np.ndarray] = []
        for i in range(n_joints):
            link_positions.append(self._dynamics.link_position(q, i))
            link_jacobians.append(self._dynamics.link_jacobian(q, i))
        # End-effector
        link_positions.append(state.ee_position)
        J_full = self._dynamics.jacobian(q)
        link_jacobians.append(J_full[:3, :n_joints])

        n_points = len(link_positions)

        # For each point, compute max barrier over all obstacles (most dangerous one)
        barrier_vals = np.zeros(n_points)
        barrier_grads: list[np.ndarray] = [np.zeros(3) for _ in range(n_points)]
        for p_idx in range(n_points):
            max_b = 0.0
            max_grad = np.zeros(3)
            for obs_center, obs_radius in obs_list:
                b = self._barrier_for_obstacle(
                    link_positions[p_idx], obs_center, obs_radius
                )
                if b > max_b:
                    max_b = b
                    max_grad = self._barrier_grad_for_obstacle(
                        link_positions[p_idx], obs_center, obs_radius
                    )
            barrier_vals[p_idx] = max_b
            barrier_grads[p_idx] = max_grad

        # Softmax weights across points
        exp_vals = np.exp(alpha * barrier_vals)
        exp_sum = exp_vals.sum()
        weights = exp_vals / exp_sum
        soft_max = np.log(exp_sum) / alpha

        # Safety control: weighted sum of barrier gradient contributions
        safety_control = np.zeros(n_joints)
        for p_idx in range(n_points):
            J_p = link_jacobians[p_idx]
            safety_control += -weights[p_idx] * J_p.T @ barrier_grads[p_idx]

        # End-effector tracking control
        ee_control = self._ee_tracking_control(q, state.ee_position, target)

        # Blend: as barrier increases, shift weight from tracking to safety
        threshold = self._k / 2 - np.log(n_points) / alpha
        if threshold <= 0:
            threshold = 1.0
        ratio = np.clip(soft_max / threshold, 0.0, 1.0)

        dq = self._velocity_scale * ee_control * (1.0 - ratio) + safety_control * ratio

        # Velocity limit clipping
        velo_limit = 2.0
        max_vel = np.abs(dq).max()
        if max_vel > velo_limit:
            dq *= velo_limit / max_vel

        barrier_value = float(soft_max)
        lyapunov_value = self._lyapunov(state.ee_position, target)

        return ControlOutput(
            command=dq,
            mode=self._mode,
            barrier_value=barrier_value,
            lyapunov_value=lyapunov_value,
            info={"soft_max_barrier": barrier_value, "blend_ratio": float(ratio)},
        )

    def _ee_tracking_control(
        self, q: np.ndarray, ee_pos: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        J = self._dynamics.jacobian(q)
        J_pos = J[:3, : PinocchioModel.ARM_DOF]
        J_pinv = np.linalg.pinv(J_pos)
        dV = self._lyapunov_grad(ee_pos, target)
        return -J_pinv @ dV

    def _tracking_only(self, state: RobotState, target: np.ndarray) -> ControlOutput:
        q = state.joint_positions
        dq = self._velocity_scale * self._ee_tracking_control(
            q, state.ee_position, target
        )
        velo_limit = 1.0
        max_vel = np.abs(dq).max()
        if max_vel > velo_limit:
            dq *= velo_limit / max_vel

        return ControlOutput(
            command=dq,
            mode=self._mode,
            barrier_value=0.0,
            lyapunov_value=self._lyapunov(state.ee_position, target),
        )

    def reset(self) -> None:
        pass

    @property
    def control_mode(self) -> ControlMode:
        return self._mode
