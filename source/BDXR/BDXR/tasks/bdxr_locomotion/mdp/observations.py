# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms specific to the BDX-R walk task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Control which observation functions are exported
__all__ = [
    "feet_contact_binary",
    "gait_phase",
    "foot_friction",
    "feet_contact_force",
    "body_mass",
]


def feet_contact_binary(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Binary per-foot ground contact, the sim analogue of the foot switches."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    return (forces.norm(dim=-1).max(dim=1)[0] > threshold).float()


def gait_phase(
        env: ManagerBasedRLEnv, command_name: str, period: float, cmd_threshold: float
) -> torch.Tensor:
    """Creates a 2D gait phase clock that we can derive reward from."""

    time_into_episode = env.episode_length_buf.float() * env.step_dt
    phi = 2 * torch.pi * time_into_episode / period
    clock = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)

    # Normalize the x y speed vector to find speed magnitude
    speed = env.command_manager.get_command(command_name)[:, :2].norm(dim=1)

    # Bool is either 1 or 0, multiply to switch clock to (0,0) standing position
    return clock * (speed > cmd_threshold).unsqueeze(-1)

def foot_friction(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mean static friction of the robot's collision shapes."""

    asset: Articulation = env.scene[asset_cfg.name]
    mats = wp.to_torch(asset.root_view.get_material_properties().to(env.device))
    # (num_envs, num_shapes, 3) -> [static_friction, dynamic_friction, restitution]
    return mats[..., 0].mean(dim=1, keepdim=True)


def feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact force magnitude on a per foot basis."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    return forces.norm(dim=-1).max(dim=1)[0]


def body_mass(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Actual mass of the input bodies, which randomize_rigid_body_mass initially disturbes."""

    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.data.body_mass.torch
    return masses[:, asset_cfg.body_ids].sum(dim=1, keepdim=True)
