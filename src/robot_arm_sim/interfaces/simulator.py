from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RobotState:
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    ee_position: np.ndarray
    ee_orientation: np.ndarray
    timestamp: float


class SimulatorInterface(ABC):
    @abstractmethod
    def setup(self, config: dict[str, Any]) -> None:
        """Initialize the simulation environment."""

    @abstractmethod
    def get_state(self) -> RobotState:
        """Return the current robot state."""

    @abstractmethod
    def apply_torques(self, torques: np.ndarray) -> None:
        """Apply joint torques to the robot."""

    @abstractmethod
    def apply_velocities(self, velocities: np.ndarray) -> None:
        """Apply joint velocities to the robot."""

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one timestep."""

    @abstractmethod
    def add_obstacle(self, position: np.ndarray, radius: float) -> int:
        """Add a spherical obstacle and return its ID."""

    @abstractmethod
    def reset(self) -> RobotState:
        """Reset the simulation to initial state."""

    @abstractmethod
    def close(self) -> None:
        """Clean up simulator resources."""
