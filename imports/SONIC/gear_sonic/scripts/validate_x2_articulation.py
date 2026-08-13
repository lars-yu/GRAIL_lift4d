#!/usr/bin/env python3
"""Load X2 in IsaacLab and verify that policy position targets move an active joint."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--test-joint", default="left_elbow_joint")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

from gear_sonic.envs.manager_env.robots.x2 import (  # noqa: E402
    X2_ACTIVE_JOINT_NAMES,
    X2_CFG,
)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    robot = Articulation(X2_CFG.replace(prim_path="/World/X2"))
    sim.reset()
    robot.update(sim_cfg.dt)

    missing = sorted(set(X2_ACTIVE_JOINT_NAMES) - set(robot.joint_names))
    if missing:
        raise RuntimeError(f"X2 USD is missing configured active joints: {missing}")
    if len(X2_ACTIVE_JOINT_NAMES) != 51 or len(set(X2_ACTIVE_JOINT_NAMES)) != 51:
        raise RuntimeError("X2 policy joint list must contain 51 unique joints")

    active_indices_list = [
        robot.joint_names.index(name) for name in X2_ACTIVE_JOINT_NAMES
    ]
    active_indices = torch.tensor(
        active_indices_list,
        dtype=torch.long,
        device=robot.device,
    )
    passive_names = [
        name for name in robot.joint_names if name not in set(X2_ACTIVE_JOINT_NAMES)
    ]
    print(f"articulation joints: {robot.num_joints}")
    print(f"policy-controlled joints: {len(active_indices)}")
    print(f"passive joints ({len(passive_names)}): {passive_names}")

    if args.test_joint not in X2_ACTIVE_JOINT_NAMES:
        raise ValueError(f"--test-joint must be active, got {args.test_joint!r}")
    test_index = robot.joint_names.index(args.test_joint)
    initial_position = robot.data.joint_pos[0, test_index].item()

    target = robot.data.default_joint_pos[:, active_indices].clone()
    target_index = X2_ACTIVE_JOINT_NAMES.index(args.test_joint)
    target[:, target_index] += 0.05
    for _ in range(args.steps):
        robot.set_joint_position_target(target, joint_ids=active_indices_list)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)

    final_position = robot.data.joint_pos[0, test_index].item()
    response = abs(final_position - initial_position)
    print(
        f"{args.test_joint}: initial={initial_position:.6f}, "
        f"final={final_position:.6f}, response={response:.6f}"
    )
    if response < 1.0e-3:
        raise RuntimeError("X2 joint did not respond to the policy position target")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
