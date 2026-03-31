from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(result_dir: str | Path) -> pd.DataFrame:
    result_dir = Path(result_dir)
    csv_path = result_dir / "data.csv"
    parquet_path = result_dir / "data.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)


def plot_joint_states(df: pd.DataFrame, save_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i in range(7):
        axes[0].plot(df["timestamp"], df[f"q_{i}"], label=f"q_{i}")
    axes[0].set_ylabel("Position [rad]")
    axes[0].set_title("Joint Positions")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i in range(7):
        axes[1].plot(df["timestamp"], df[f"dq_{i}"], label=f"dq_{i}")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Velocity [rad/s]")
    axes[1].set_title("Joint Velocities")
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "joint_states.png", dpi=150)
    plt.close(fig)


def plot_task_space_trajectory(df: pd.DataFrame, save_dir: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(df["ee_x"], df["ee_y"], df["ee_z"], "b-", linewidth=1.5, label="EE Trajectory")
    ax.scatter(
        df["target_x"].iloc[0],
        df["target_y"].iloc[0],
        df["target_z"].iloc[0],
        c="g", s=100, marker="*", label="Target",
    )
    ax.scatter(
        df["ee_x"].iloc[0],
        df["ee_y"].iloc[0],
        df["ee_z"].iloc[0],
        c="r", s=60, marker="o", label="Start",
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("End-Effector Trajectory in Task Space")
    ax.legend()

    fig.savefig(save_dir / "task_space_trajectory.png", dpi=150)
    plt.close(fig)


def plot_safety_indicators(df: pd.DataFrame, save_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(df["timestamp"], df["barrier_value"], "r-", linewidth=1.5)
    axes[0].set_ylabel("Barrier Value")
    axes[0].set_title("Barrier Function (Safety)")
    axes[0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["timestamp"], df["lyapunov_value"], "b-", linewidth=1.5)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Lyapunov Value")
    axes[1].set_title("Lyapunov Function (Convergence)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "safety_indicators.png", dpi=150)
    plt.close(fig)


def plot_control_commands(df: pd.DataFrame, save_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(7):
        ax.plot(df["timestamp"], df[f"cmd_{i}"], label=f"cmd_{i}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Command")
    ax.set_title("Control Commands")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_dir / "control_commands.png", dpi=150)
    plt.close(fig)


def generate_report(result_dir: str | Path) -> None:
    result_dir = Path(result_dir)
    df = load_data(result_dir)
    plots_dir = result_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid")
    plot_joint_states(df, plots_dir)
    plot_task_space_trajectory(df, plots_dir)
    plot_safety_indicators(df, plots_dir)
    plot_control_commands(df, plots_dir)
    print(f"Report generated: {plots_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m robot_arm_sim.visualization.analytics <result_dir>")
        sys.exit(1)
    generate_report(sys.argv[1])
