from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(result_dir: Path) -> pd.DataFrame:
    parquet_path = result_dir / "data.parquet"
    csv_path = result_dir / "data.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)


class PlotViewer:
    """matplotlib GUI로 시뮬레이션 결과를 탭 형태로 탐색한다."""

    PAGE_NAMES = [
        "Joint States",
        "Task Space Trajectory",
        "Safety Indicators",
        "Control Commands",
    ]

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._page = 0

        self._fig = plt.figure(figsize=(13, 8))
        self._fig.canvas.manager.set_window_title("Simulation Result Viewer")

        # 하단 버튼 영역 확보
        self._fig.subplots_adjust(bottom=0.12)
        ax_prev = self._fig.add_axes([0.3, 0.02, 0.1, 0.04])
        ax_next = self._fig.add_axes([0.6, 0.02, 0.1, 0.04])
        self._btn_prev = Button(ax_prev, "< Prev")
        self._btn_next = Button(ax_next, "Next >")
        self._btn_prev.on_clicked(self._on_prev)
        self._btn_next.on_clicked(self._on_next)

        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw_page()
        plt.show()

    # ---- navigation ----

    def _on_prev(self, _event) -> None:
        self._page = (self._page - 1) % len(self.PAGE_NAMES)
        self._draw_page()

    def _on_next(self, _event) -> None:
        self._page = (self._page + 1) % len(self.PAGE_NAMES)
        self._draw_page()

    def _on_key(self, event) -> None:
        if event.key == "right":
            self._on_next(event)
        elif event.key == "left":
            self._on_prev(event)

    # ---- drawing ----

    def _clear_axes(self) -> None:
        for ax in self._fig.axes:
            if ax not in (self._btn_prev.ax, self._btn_next.ax):
                ax.remove()

    def _draw_page(self) -> None:
        self._clear_axes()
        draw_fn = [
            self._draw_joint_states,
            self._draw_task_space,
            self._draw_safety,
            self._draw_commands,
        ][self._page]
        draw_fn()
        self._fig.suptitle(
            f"[{self._page + 1}/{len(self.PAGE_NAMES)}] {self.PAGE_NAMES[self._page]}",
            fontsize=13,
            fontweight="bold",
        )
        self._fig.canvas.draw_idle()

    def _draw_joint_states(self) -> None:
        df = self._df
        ax0 = self._fig.add_subplot(2, 1, 1)
        ax1 = self._fig.add_subplot(2, 1, 2, sharex=ax0)
        self._fig.subplots_adjust(bottom=0.12, top=0.90, hspace=0.35)

        for i in range(7):
            ax0.plot(df["timestamp"], df[f"q_{i}"], label=f"q_{i}")
        ax0.set_ylabel("Position [rad]")
        ax0.set_title("Joint Positions")
        ax0.legend(ncol=4, fontsize=8)
        ax0.grid(True, alpha=0.3)

        for i in range(7):
            ax1.plot(df["timestamp"], df[f"dq_{i}"], label=f"dq_{i}")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Velocity [rad/s]")
        ax1.set_title("Joint Velocities")
        ax1.legend(ncol=4, fontsize=8)
        ax1.grid(True, alpha=0.3)

    def _draw_task_space(self) -> None:
        df = self._df
        ax = self._fig.add_subplot(111, projection="3d")
        self._fig.subplots_adjust(bottom=0.12, top=0.90)

        ax.plot(df["ee_x"], df["ee_y"], df["ee_z"], "b-", linewidth=1.5, label="EE Trajectory")
        ax.scatter(
            df["target_x"].iloc[0], df["target_y"].iloc[0], df["target_z"].iloc[0],
            c="g", s=100, marker="*", label="Target",
        )
        ax.scatter(
            df["ee_x"].iloc[0], df["ee_y"].iloc[0], df["ee_z"].iloc[0],
            c="r", s=60, marker="o", label="Start",
        )
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.legend()

    def _draw_safety(self) -> None:
        df = self._df
        ax0 = self._fig.add_subplot(2, 1, 1)
        ax1 = self._fig.add_subplot(2, 1, 2, sharex=ax0)
        self._fig.subplots_adjust(bottom=0.12, top=0.90, hspace=0.35)

        ax0.plot(df["timestamp"], df["barrier_value"], "r-", linewidth=1.5)
        ax0.set_ylabel("Barrier Value")
        ax0.set_title("Barrier Function (Safety)")
        ax0.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax0.grid(True, alpha=0.3)

        ax1.plot(df["timestamp"], df["lyapunov_value"], "b-", linewidth=1.5)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Lyapunov Value")
        ax1.set_title("Lyapunov Function (Convergence)")
        ax1.grid(True, alpha=0.3)

    def _draw_commands(self) -> None:
        df = self._df
        ax = self._fig.add_subplot(111)
        self._fig.subplots_adjust(bottom=0.12, top=0.90)

        for i in range(7):
            ax.plot(df["timestamp"], df[f"cmd_{i}"], label=f"cmd_{i}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Command")
        ax.legend(ncol=4, fontsize=8)
        ax.grid(True, alpha=0.3)


def find_latest_result(base_dir: Path = Path("results")) -> Path:
    """results/ 내에서 가장 최근에 생성된 결과 디렉토리를 반환한다."""
    if not base_dir.is_dir():
        print(f"Error: results directory '{base_dir}' not found")
        sys.exit(1)

    subdirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not subdirs:
        print(f"Error: no result directories found in '{base_dir}'")
        sys.exit(1)

    return subdirs[0]


def main() -> None:
    if len(sys.argv) >= 2:
        result_dir = Path(sys.argv[1])
    else:
        result_dir = find_latest_result()
        print(f"No argument provided. Using latest result: {result_dir}")

    if not result_dir.is_dir():
        print(f"Error: '{result_dir}' is not a directory")
        sys.exit(1)

    df = load_data(result_dir)
    sns.set_theme(style="whitegrid")
    PlotViewer(df)


if __name__ == "__main__":
    main()
