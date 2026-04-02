from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config.manager import ConfigManager

# Import controllers and other components
from .controller.clbf_controller import CLBFController
from .controller.pd_controller import PDController
from .controller.robust_cbf_controller import RobustCBFController

# data engine for recording simulation data
from .data.engine import DataEngine

# Pinocchio model for robot dynamics
from .dynamics.pinocchio_model import PinocchioModel

# interfaces
from .interfaces.controller import ControlMode, ControllerInterface
from .interfaces.simulator import SimulatorInterface

# simulator implementation
from .simulator.pybullet_sim import PyBulletSimulator

# analysis and visualization
from .visualization.analytics import generate_report


def build_pinocchio_model(config: dict) -> PinocchioModel:
    import pybullet_data

    urdf_path = str(Path(pybullet_data.getDataPath()) / config["robot"]["urdf"])
    return PinocchioModel(urdf_path)


def run_simulation(
    simulator: SimulatorInterface,
    controller: ControllerInterface,
    data_engine: DataEngine,
    config: dict,
) -> None:
    sim_cfg = config["simulation"]
    dt = sim_cfg["timestep"]
    duration = sim_cfg["duration"]
    ctrl_freq = sim_cfg["control_frequency"]
    sim_steps_per_ctrl = max(1, int(1.0 / (dt * ctrl_freq)))

    target = np.array(config["target"]["ee_position"], dtype=float)
    obstacles = config.get("obstacles", [])

    total_steps = int(duration / dt)
    ctrl_step = 0

    state = simulator.get_state()
    control = controller.compute(state, target, obstacles)

    for step in range(total_steps):
        # Control update at specified frequency
        if step % sim_steps_per_ctrl == 0:
            state = simulator.get_state()
            control = controller.compute(state, target, obstacles)
            data_engine.record(state, control, target)
            ctrl_step += 1
            simulator.update_link_spheres()

        # Apply control
        if controller.control_mode == ControlMode.TORQUE:
            simulator.apply_torques(control.command)
        else:
            simulator.apply_velocities(control.command)

        simulator.step()

    # Record final state
    state = simulator.get_state()
    control = controller.compute(state, target, obstacles)
    data_engine.record(state, control, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot Arm CLBF Simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    # Load config
    cfg_manager = ConfigManager(args.config)
    config = cfg_manager.config

    # Build components
    dynamics = build_pinocchio_model(config)
    simulator = PyBulletSimulator()
    ctrl_type = config.get("controller", {}).get("type", "pd")
    if ctrl_type == "clbf":
        controller: ControllerInterface = CLBFController(dynamics)
    elif ctrl_type == "robust_cbf":
        controller = RobustCBFController(dynamics)
    else:
        controller = PDController(dynamics)
    data_engine = DataEngine()

    # Setup
    simulator.setup(config)
    controller.setup(config)

    # Visualise link safety spheres if configured
    ctrl_cfg = config.get("controller", {})
    rcbf = ctrl_cfg.get("robust_cbf", {})
    clbf = ctrl_cfg.get("clbf", {})

    if "link_spheres" in rcbf:
        sphere_specs = [
            {
                "link": s["link"],
                "offset": s.get("offset", [0, 0, 0]),
                "radius": s["radius"],
            }
            for s in rcbf["link_spheres"]
        ]
        simulator.setup_link_spheres(sphere_specs)
    elif "link_radii" in rcbf:
        sphere_specs = [
            {"link": i, "offset": [0, 0, 0], "radius": r}
            for i, r in enumerate(rcbf["link_radii"])
        ]
        simulator.setup_link_spheres(sphere_specs)
    elif "link_radii" in clbf:
        sphere_specs = [
            {"link": i, "offset": [0, 0, 0], "radius": r}
            for i, r in enumerate(clbf["link_radii"])
        ]
        simulator.setup_link_spheres(sphere_specs)

    print(f"Simulation starting: duration={config['simulation']['duration']}s")
    t0 = time.time()
    run_simulation(simulator, controller, data_engine, config)
    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed:.2f}s (wall-clock)")

    # Save results
    output_dir = config["data"]["output_dir"]
    fmt = config["data"].get("format", "csv")
    result_dir = data_engine.save(output_dir, fmt=fmt)
    cfg_manager.save_to(result_dir)
    print(f"Data saved to: {result_dir}")

    # Generate analysis report
    generate_report(result_dir)

    simulator.close()


if __name__ == "__main__":
    main()
