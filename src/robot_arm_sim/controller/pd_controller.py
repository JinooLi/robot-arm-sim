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


class PDController(ControllerInterface):
    """Task-space PD controller with gravity compensation.

    This is a reference implementation showing how to use the controller
    interface. Users can implement their own CLBF controller by subclassing
    ControllerInterface and computing barrier/lyapunov values.
    """

    def __init__(self, dynamics: PinocchioModel) -> None:
        self._dynamics = dynamics
        self._kp = np.ones(7) * 100.0
        self._kd = np.ones(7) * 20.0
        self._mode = ControlMode.TORQUE

    def setup(self, config: dict[str, Any]) -> None:
        ctrl_cfg = config["controller"]
        self._kp = np.array(ctrl_cfg["gains"]["kp"], dtype=float)
        self._kd = np.array(ctrl_cfg["gains"]["kd"], dtype=float)
        mode_str = ctrl_cfg.get("control_mode", "torque")
        self._mode = (
            ControlMode.TORQUE if mode_str == "torque" else ControlMode.VELOCITY
        )

    def compute(
        self,
        state: RobotState,
        target: np.ndarray,
        obstacles: list[dict[str, Any]],
    ) -> ControlOutput:
        q = state.joint_positions
        dq = state.joint_velocities

        # Current end-effector position
        ee_pos = self._dynamics.forward_kinematics(q)
        J = self._dynamics.jacobian(q)
        J_pos = J[:3, :7]  # position Jacobian

        # Task-space error
        x_err = target - ee_pos

        if self._mode == ControlMode.TORQUE:
            g = self._dynamics.gravity_vector(q)
            # Task-space PD + gravity compensation
            # Map task-space force to joint torques via J^T
            f_task = self._kp[:3] * x_err - self._kd[:3] * (J_pos @ dq)
            tau = J_pos.T @ f_task + g[:7]
            command = tau
        else:
            # Velocity mode: simple proportional control
            dq_des = np.linalg.pinv(J_pos) @ (self._kp[:3] * x_err)
            command = dq_des

        return ControlOutput(
            command=command,
            mode=self._mode,
            barrier_value=0.0,
            lyapunov_value=float(np.dot(x_err, x_err)),
            info={"ee_position": ee_pos, "task_error": x_err},
        )

    def reset(self) -> None:
        pass

    @property
    def control_mode(self) -> ControlMode:
        return self._mode
