from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from .simulator import RobotState


class ControlMode(Enum):
    VELOCITY = auto()
    TORQUE = auto()


@dataclass
class ControlOutput:
    command: np.ndarray
    mode: ControlMode
    barrier_value: float = 0.0
    lyapunov_value: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


class ControllerInterface(ABC):
    @abstractmethod
    def setup(self, config: dict[str, Any]) -> None:
        """Initialize the controller with configuration."""

    @abstractmethod
    def compute(
        self,
        state: RobotState,
        target: np.ndarray,
        obstacles: list[dict[str, Any]],
    ) -> ControlOutput:
        """Compute the control output given the current state, target, and obstacles."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the controller internal state."""

    @property
    @abstractmethod
    def control_mode(self) -> ControlMode:
        """Return the control mode this controller uses."""
