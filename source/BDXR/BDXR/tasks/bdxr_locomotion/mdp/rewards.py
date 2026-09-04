# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms specific to the BDX-R walk task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Control which reward functions actually get exported
__all__ = [
    "body_xy_ang_acc_l2",
    "base_height_floor_l2",
    "gait_contact_schedule",
    "foot_swing_height_track",
    "get_phase",
    "head_attitude_tracking_exp",
    "head_yaw_tracking_exp"
]

def body_xy_ang_acc_l2(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
    """Penalize the pitch roll angular acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.linalg.norm(asset.data.body_ang_acc_w.torch[:, asset_cfg.body_ids, :2], dim=-1), dim=1)

def base_height_floor_l2(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_Foot"),
) -> torch.Tensor:
    """Penalize the base height being below a minimum height."""

    asset: Articulation = env.scene[asset_cfg.name]

    # Minimum takes position of lowest foot
    foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2].min(dim=1)[0]
    height_offset = asset.data.root_pos_w.torch[:, 2] - foot_z

    # Quadratic makes it softer initially, then sharper as the robot sags lower
    return torch.clamp(minimum_height - height_offset, min=0.0).square()


def gait_contact_schedule(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    period: float,
    duty: float,
    cmd_threshold: float,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the feet matching the alternating stance schedule."""

    phase = get_phase(env, period)

    # Each foot stands for the first `duty` of its own cycle; foot 1 trails by half a cycle
    want = torch.stack([(phase < duty).float(), (((phase + 0.5) % 1.0) < duty).float()], dim=-1)

    # Max over the sensor history debounces single-frame contact dropouts
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    have = (forces.norm(dim=-1).max(dim=1)[0] > force_threshold).float()

    # Product, not mean: under a mean, standing on both feet scores 0.5 for free
    reward = (1.0 - (want - have).abs()).prod(dim=-1)

    # Clock never stops, so gate off at zero command or this pays the robot to march in place
    speed = env.command_manager.get_command(command_name)[:, :2].norm(dim=1)
    return reward * (speed > cmd_threshold)


def foot_swing_height_track(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period: float,
    duty: float,
    swing_height: float,
    ref_speed: float,
    std: float,
    cmd_threshold: float,
) -> torch.Tensor:
    """Track a phase indexed half swing profile on each foot, scaled by commanded speed."""

    phase = get_phase(env, period)

    def swing_shape(phase: torch.Tensor) -> torch.Tensor:
        """Unit half-sine over the swing window; the clamp flattens stance to zero."""
        return torch.sin(torch.pi * ((phase - duty) / (1.0 - duty)).clamp(0.0, 1.0))

    # Same profile sampled gait offset difference, half a cycle, apart. Column order
    # matches gait_contact_schedule so both rewards agree on which foot is swinging
    shape = torch.stack([swing_shape(phase), swing_shape((phase + 0.5) % 1.0)], dim=-1)

    # Amplitude shrinks at low speed, so there's a creep at lower speeds
    speed = env.command_manager.get_command(command_name)[:, :2].norm(dim=1)
    want = shape * (swing_height * (speed / ref_speed).clamp(0.25, 1.0)).unsqueeze(-1)

    # Height above the lower foot, so this stays terrain-agnostic without a ray-caster
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    have = foot_z - foot_z.min(dim=1, keepdim=True)[0]

    # Gaussian falloff, std leaves gradient
    reward = torch.exp(-(have - want).square().sum(dim=-1) / std**2)

    return reward * (speed > cmd_threshold)


def get_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """Position within the current gait cycle, in [0, 1)."""

    return ((env.episode_length_buf.float() * env.step_dt) / period) % 1.0

def head_attitude_tracking_exp(
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        std: float = 0.2,
) -> torch.Tensor:
    """Head reward for holding commanded gravity referenced pitch/roll"""

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get gravity from the head reference frame by applying an 
    # inverse quat rotation to it.
    gravity_head = quat_apply_inverse(
        asset.data.body_quat_w.torch[:, asset_cfg.body_ids[0], :],
        asset.data.GRAVITY_VEC_W.torch
    )

    # Get pitch by applying an asin on (head_relative_x_grav_component / real_world_grav)
    # Real world grav is a unit vector so it cancels
    pitch = torch.asin(gravity_head[:, 0].clamp(-1, 1))

    # Get roll by applying an atan2 on the negatives of the head relative y and z components of gravity
    # Using these two components silently absorbs any coupling from pitch joints
    roll = torch.atan2(-gravity_head[:, 1], -gravity_head[:, 2])

    # Square individual pitch and roll reward components to reduce reward coupling
    error = (pitch - command[:, 0]).square() + (roll - command[:, 1]).square()

    # Gaussian Kernel on the squared error
    return torch.exp(-error / std ** 2)

def head_yaw_tracking_exp(
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        std: float = 0.5
) -> torch.Tensor:
    """Reward the head for holding the commanded body relative yaw tensor
    
    This is held different from the head_attitude_tracking_exp rew function
    because this is something that slews to the position over time. It is
    a different, less immediate motion that the attitude, thus it gets a different
    reward function.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get body relative yaw, raw joint position of head_yaw
    yaw = - asset.data.joint_pos.torch[:, asset_cfg.joint_ids[0]]
    error = (yaw - command[:, 2]).square()

    # Guassian kernel on the squared error
    return torch.exp(-error / std ** 2)
