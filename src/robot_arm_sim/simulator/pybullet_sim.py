from __future__ import annotations

import time
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data

from ..interfaces.simulator import RobotState, SimulatorInterface


class PyBulletSimulator(SimulatorInterface):
    def __init__(self) -> None:
        self._client: int | None = None
        self._robot_id: int | None = None
        self._joint_indices: list[int] = []
        self._obstacle_ids: list[int] = []
        self._timestep: float = 0.001
        self._sim_time: float = 0.0
        self._initial_q: np.ndarray = np.zeros(7)
        self._realtime: bool = False
        self._wall_start: float = 0.0

    def setup(self, config: dict[str, Any]) -> None:
        sim_cfg = config["simulation"]
        robot_cfg = config["robot"]
        target_cfg = config["target"]

        is_gui = sim_cfg["mode"].upper() == "GUI"
        self._client = p.connect(p.GUI if is_gui else p.DIRECT)
        self._realtime = is_gui
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self._timestep = sim_cfg["timestep"]
        p.setTimeStep(self._timestep)
        p.setGravity(*sim_cfg["gravity"])

        p.loadURDF("plane.urdf")
        self._robot_id = p.loadURDF(
            robot_cfg["urdf"],
            basePosition=[0, 0, 0],
            useFixedBase=True,
        )

        # Identify the 7 revolute joints (skip fixed joints)
        self._joint_indices = []
        for i in range(p.getNumJoints(self._robot_id)):
            info = p.getJointInfo(self._robot_id, i)
            if info[2] == p.JOINT_REVOLUTE:
                self._joint_indices.append(i)
            if len(self._joint_indices) == robot_cfg["num_joints"]:
                break

        self._initial_q = np.array(robot_cfg["initial_joint_positions"], dtype=float)
        self._set_joint_positions(self._initial_q)

        # Draw target position for visualization
        self.draw_debug_point(position=np.array(target_cfg["ee_position"]))

        # Add obstacles from config
        for obs in config.get("obstacles", []):
            self.add_obstacle(
                np.array(obs["position"]),
                obs["radius"],
                obs.get("color", [1, 0, 0, 0.5]),
            )

        # Disable default velocity controllers
        for idx in self._joint_indices:
            p.setJointMotorControl2(
                self._robot_id,
                idx,
                p.VELOCITY_CONTROL,
                force=0,
            )

        self._sim_time = 0.0
        self._wall_start = time.perf_counter()

    def _set_joint_positions(self, q: np.ndarray) -> None:
        for i, idx in enumerate(self._joint_indices):
            p.resetJointState(self._robot_id, idx, q[i], 0.0)

    def get_state(self) -> RobotState:
        positions = []
        velocities = []
        for idx in self._joint_indices:
            state = p.getJointState(self._robot_id, idx)
            positions.append(state[0])
            velocities.append(state[1])

        # End-effector: use link index 11 (panda_hand)
        ee_state = p.getLinkState(self._robot_id, 11)
        return RobotState(
            joint_positions=np.array(positions),
            joint_velocities=np.array(velocities),
            ee_position=np.array(ee_state[0]),
            ee_orientation=np.array(ee_state[1]),
            timestamp=self._sim_time,
        )

    def apply_torques(self, torques: np.ndarray) -> None:
        for i, idx in enumerate(self._joint_indices):
            p.setJointMotorControl2(
                self._robot_id,
                idx,
                p.TORQUE_CONTROL,
                force=float(torques[i]),
            )

    def apply_velocities(self, velocities: np.ndarray) -> None:
        for i, idx in enumerate(self._joint_indices):
            p.setJointMotorControl2(
                self._robot_id,
                idx,
                p.VELOCITY_CONTROL,
                targetVelocity=float(velocities[i]),
                force=50.0,
            )

    def step(self) -> None:
        p.stepSimulation()
        self._sim_time += self._timestep
        if self._realtime:
            wall_elapsed = time.perf_counter() - self._wall_start
            sleep_time = self._sim_time - wall_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def add_obstacle(
        self,
        position: np.ndarray,
        radius: float,
        color: list[float] | None = None,
    ) -> int:
        if color is None:
            color = [1, 0, 0, 0.5]
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
        )
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position.tolist(),
        )
        self._obstacle_ids.append(body_id)
        return body_id

    def reset(self) -> RobotState:
        self._set_joint_positions(self._initial_q)
        self._sim_time = 0.0
        self._wall_start = time.perf_counter()
        return self.get_state()

    def close(self) -> None:
        if self._client is not None:
            p.disconnect(self._client)
            self._client = None

    def draw_debug_point(self, position, color=[1, 0, 0], size=1, lifeTime=0):
        """
        position: [x, y, z] 좌표
        color: [r, g, b] (0~1 사이 값)
        size: 십자가 크기
        lifeTime: 유지 시간 (0이면 영구 유지, 양수면 초 단위 후 사라짐)
        """
        x, y, z = position

        # X축 선
        p.addUserDebugLine(
            [x - size, y, z], [x + size, y, z], lineColorRGB=color, lifeTime=lifeTime
        )
        # Y축 선
        p.addUserDebugLine(
            [x, y - size, z], [x, y + size, z], lineColorRGB=color, lifeTime=lifeTime
        )
        # Z축 선
        p.addUserDebugLine(
            [x, y, z - size], [x, y, z + size], lineColorRGB=color, lifeTime=lifeTime
        )
