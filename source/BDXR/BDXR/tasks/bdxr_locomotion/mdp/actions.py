# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms specific to the BDX-R walk task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

__all__ = ["RateLimitedJointPositionAction", "RateLimitedJointPositionActionCfg"]


class RateLimitedJointPositionAction(JointPositionAction):
    """Joint position action whose target may not move faster than the servo can.

    A hard constraint, so the commanded trajectory is always something the PD loop can
    follow and the actuator never sees a step input it will just saturate against.
    """

    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        # Seed at the default pose -- the same value _offset resolves to when
        # use_default_offset is True, so step 0 starts with zero rate demand
        self._last_target = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        # Runs once per policy step, which is the rate the limit is defined against;
        # apply_actions() runs `decimation` times per step and would clamp repeatedly
        super().process_actions(actions)

        limit = self.cfg.max_joint_velocity * self._env.step_dt
        self._processed_actions = torch.clamp(
            self._processed_actions,
            self._last_target - limit,
            self._last_target + limit,
        )
        self._last_target = self._processed_actions.clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Without this, a freshly reset robot inherits the previous episode's target and
        # the limiter drags it back from a stale value -- a transient every episode
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._last_target[env_ids] = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids]


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`RateLimitedJointPositionAction`."""

    class_type: type[ActionTerm] = RateLimitedJointPositionAction

    max_joint_velocity: float = MISSING
    """Max rate of change of the position target, rad/s.

    Set to the slowest actuated joint's velocity_limit_sim so the limiter and the
    physics agree on what the servo can do.
    """
