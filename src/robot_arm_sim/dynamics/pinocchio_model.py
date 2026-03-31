from __future__ import annotations

from typing import Any

import numpy as np
import pinocchio as pin


class PinocchioModel:
    """Wrapper around Pinocchio for computing Jacobians and dynamics matrices.

    The Panda URDF has 9 joints (7 arm + 2 finger). This wrapper handles
    padding: all public methods accept 7-DOF arm-only vectors and return
    7-DOF results.
    """

    ARM_DOF = 7

    def __init__(self, urdf_path: str) -> None:
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nq_full = self.model.nq
        self.nv_full = self.model.nv
        self.ee_frame_id = self.model.getFrameId("panda_hand")

    def _pad_q(self, q7: np.ndarray) -> np.ndarray:
        q_full = np.zeros(self.nq_full)
        q_full[: self.ARM_DOF] = q7
        return q_full

    def _pad_v(self, v7: np.ndarray) -> np.ndarray:
        v_full = np.zeros(self.nv_full)
        v_full[: self.ARM_DOF] = v7
        return v_full

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        q_full = self._pad_q(q)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_frame_id].translation.copy()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q_full = self._pad_q(q)
        pin.computeJointJacobians(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(
            self.model, self.data, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        # Return only the 7 arm-joint columns
        return J[:, : self.ARM_DOF].copy()

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        q_full = self._pad_q(q)
        M = pin.crba(self.model, self.data, q_full)
        return M[: self.ARM_DOF, : self.ARM_DOF].copy()

    def coriolis(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        q_full = self._pad_q(q)
        dq_full = self._pad_v(dq)
        C = pin.computeCoriolisMatrix(self.model, self.data, q_full, dq_full)
        return C[: self.ARM_DOF, : self.ARM_DOF].copy()

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        q_full = self._pad_q(q)
        g = pin.computeGeneralizedGravity(self.model, self.data, q_full)
        return g[: self.ARM_DOF].copy()

    def nonlinear_effects(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        q_full = self._pad_q(q)
        dq_full = self._pad_v(dq)
        nle = pin.nonLinearEffects(self.model, self.data, q_full, dq_full)
        return nle[: self.ARM_DOF].copy()

    def link_position(self, q: np.ndarray, joint_idx: int) -> np.ndarray:
        """Return the world-frame position of a joint/link (0-indexed arm joint)."""
        q_full = self._pad_q(q)
        pin.forwardKinematics(self.model, self.data, q_full)
        # joint_idx 0..6 -> pinocchio joint 1..7 (joint 0 is universe)
        return self.data.oMi[joint_idx + 1].translation.copy()

    def link_jacobian(self, q: np.ndarray, joint_idx: int) -> np.ndarray:
        """Return the 3xARM_DOF linear Jacobian for a joint/link (0-indexed)."""
        q_full = self._pad_q(q)
        pin.computeJointJacobians(self.model, self.data, q_full)
        J = pin.getJointJacobian(
            self.model, self.data, joint_idx + 1, pin.LOCAL_WORLD_ALIGNED
        )
        return J[:3, : self.ARM_DOF].copy()
