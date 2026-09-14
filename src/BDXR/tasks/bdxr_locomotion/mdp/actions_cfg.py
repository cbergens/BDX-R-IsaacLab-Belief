# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term configs for the BDX-R walk task."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .actions import RateLimitedJointPositionAction

__all__ = ["RateLimitedJointPositionActionCfg"]


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`RateLimitedJointPositionAction`."""

    class_type: type[RateLimitedJointPositionAction] | str = "{DIR}.actions:RateLimitedJointPositionAction"

    max_joint_velocity: float = MISSING
    """Max rate of change of the position target, rad/s.

    Set to the slowest actuated joint's velocity_limit_sim so the limiter and the
    physics agree on what the servo can do.
    """