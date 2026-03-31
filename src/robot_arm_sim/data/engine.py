from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..interfaces.controller import ControlOutput
from ..interfaces.simulator import RobotState


class DataEngine:
    """Collects simulation data into a DataFrame and saves to disk."""

    COLUMNS = [
        "timestamp",
        "q_0", "q_1", "q_2", "q_3", "q_4", "q_5", "q_6",
        "dq_0", "dq_1", "dq_2", "dq_3", "dq_4", "dq_5", "dq_6",
        "ee_x", "ee_y", "ee_z",
        "cmd_0", "cmd_1", "cmd_2", "cmd_3", "cmd_4", "cmd_5", "cmd_6",
        "barrier_value", "lyapunov_value",
        "target_x", "target_y", "target_z",
    ]

    def __init__(self) -> None:
        self._records: list[dict[str, float]] = []

    def record(
        self,
        state: RobotState,
        control: ControlOutput,
        target: np.ndarray,
    ) -> None:
        row: dict[str, float] = {"timestamp": state.timestamp}

        for i in range(7):
            row[f"q_{i}"] = state.joint_positions[i]
            row[f"dq_{i}"] = state.joint_velocities[i]
            row[f"cmd_{i}"] = control.command[i]

        row["ee_x"] = state.ee_position[0]
        row["ee_y"] = state.ee_position[1]
        row["ee_z"] = state.ee_position[2]
        row["barrier_value"] = control.barrier_value
        row["lyapunov_value"] = control.lyapunov_value
        row["target_x"] = target[0]
        row["target_y"] = target[1]
        row["target_z"] = target[2]

        self._records.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records, columns=self.COLUMNS)

    def save(
        self,
        output_dir: str | Path,
        fmt: str = "csv",
        tag: str | None = None,
    ) -> Path:
        if tag is None:
            tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = Path(output_dir) / tag
        dest.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        if fmt == "parquet":
            filepath = dest / "data.parquet"
            df.to_parquet(filepath, index=False)
        else:
            filepath = dest / "data.csv"
            df.to_csv(filepath, index=False)

        return dest

    def reset(self) -> None:
        self._records.clear()
