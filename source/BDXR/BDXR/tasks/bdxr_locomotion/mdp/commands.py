"""Command terms specific to the Bdxr-Walk-v0 task."""

from __future__ import annotations

import torch                                                                        
from collections.abc import Sequence                                                
from dataclasses import MISSING                                                     
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class HeadPoseCommand(CommandTerm):
    """Absolute head pose set points for external control

    Emits (pitch, roll, yaw) vector with pitch and roll IMU gravity-framed 
    and yaw body-framed
    """

    def __init__(self, cfg: HeadPoseCommandCfg, env: ManagerBasedRLEnv):
        """ Initialization function initializing self's data tensors and robot articulation IDs"""

        super().__init__(cfg, env)

        # Define robot articulation,  _yaw_joint_id, and _head_id
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._yaw_joint_id, _ = self.robot.find_joints(cfg.yaw_joint_name)
        self._head_id, _ = self.robot.find_bodies(cfg.head_body_name)

        # Ensure only one head joint is found
        assert len(self._head_id) == 1, f'head_body_name, "{cfg.head_body_name}," matched {len(self._head_id)} bodies'

        # Init pos [num_envs, 3] dimensions command tensor carrying (pitch, roll, yaw) for each env
        self.pose_command = torch.zeros(self.num_envs, 3, device=self.device)

        # Init 1D tensor along num_envs of bools indicative of correlated env holding a still command
        self.envs_still = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Init head_tilt_metrics, all 1D along num_envs
        self.metrics["head_tilt"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["attitude_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["yaw_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Returns a string providing head pose command information"""

        return f"HeadPoseCommand: body={self.cfg.head_body_name}, resample={self.cfg.resampling_time_range} s"
    
    @property
    def command(self) -> torch.Tensor:
        """Head_pos_target, [num_envs, 3] shaped tesnor containing pitch, roll, yaw per env"""

        return self.pose_command

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample command for training command resampling."""

        # Create temporary 1D tensor buffer between
        # self.cfg.ranges.x_affilitated_command_param (pitch, roll, or yaw) and self.pose_command[env_ids, x]
        r = torch.empty(len(env_ids), device=self.device)

        # Place pitch roll and yaw into their contracted idxs within pose_command
        self.pose_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.pitch)
        self.pose_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.roll)
        self.pose_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.yaw)

        # Assign still envs based on bernoulli draw across uniform distribution
        self.envs_still[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_still_envs

    def _update_command(self):
        """Function to zero still commands"""

        self.pose_command[self.envs_still] = 0.0

    def _update_metrics(self):
        """Update gathered metrics of the head"""

        # Original Shape is (num_instances, num_bodies, 4), indexed to only include _head_id
        head_quat = self.robot.data.body_quat_w.torch[:, self._head_id[0], :]

        # Apply inverse quaternion rotation to GRAVITY_VEC_W, rotating the world-frame gravity
        # vector relative to the head's position.
        grav_head = quat_apply_inverse(head_quat, self.robot.data.GRAVITY_VEC_W.torch)

        # Figure out the degree of tilt by taking the atan of the head-relative (0, 0, 1) and world relative (0, 0, 1)
        tilt = torch.atan2(grav_head[:, :2].norm(dim=-1), -grav_head[:, 2])

        # Find pitch by taking the asin of the real world x component of the head's (0, 0, 1) and its relative (0, 0, 1).
        pitch = torch.asin(grav_head[:, 0].clamp(-1, 1))

        # Find roll by taking the atan2 of the head frame y and z components of the real world's (0, 0, -1) gravity vector
        # NOTE: roll uses atan2 because it is the closest to the head and must absorb any coupling from pitch control
        roll = torch.atan2(-grav_head[:, 1], -grav_head[:, 2])

        yaw = self.robot.data.joint_pos.torch[:, self._yaw_joint_id[0]]

        # Gather error by stacking and normalizing individual pitch, roll, yaw error
        attitude_error = torch.stack(
            [pitch - self.pose_command[:, 0], roll - self.pose_command[:, 1]], dim=-1
        ).norm(dim=-1)

        # Accumulate episodic mean error to account for zeroing on reset
        self.metrics["head_tilt"] += tilt / self._env.max_episode_length
        self.metrics["attitude_error"] += attitude_error / self._env.max_episode_length
        self.metrics["yaw_error"] += (yaw - self.pose_command[:, 2]).abs() / self._env.max_episode_length

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Makes debugging arrows visible in the sim"""

        # If no visualizer is present and a visualization is desired, make one
        if debug_vis and not hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)

        # If there is already a visualizer, set the visability to debug_vis
        if hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        """Runs off post update stream, outliving articulations"""

        # Do nothing if robot is not initialized
        if not self.robot.is_initialized:
            return
        
        target_quat = quat_mul(

            # World yaw
            yaw_quat(self.robot.data.root_quat_w),

            # Pose command pitch, roll, yaw
            quat_from_euler_xyz(
                self.pose_command[:, 1],
                self.pose_command[:, 0],
                self.pose_command[:, 2],
            ),
        )

        # Write the arrows to the visualizer
        self.goal_pose_visualizer.visualize(
            self.robot.data.body_pos_w.torch[:, self._head_id[0]], target_quat
        )

@configclass
class HeadPoseCommandCfg(CommandTermCfg):
    """Config for head pose command"""

    class_type: type = HeadPoseCommand
    asset_name: str = MISSING
    head_body_name: str = "head"
    yaw_joint_name: str = "head_yaw"

    # Fraction of environments with a still head position to hold steady
    rel_still_envs: float = 0.25

    # Set red arrow to track the command vector prim
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path = "/Visuals/Command/head_pose_goal"
    )

    # Scale down arrow to keep it small
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.25, 0.25, 0.25)

    @configclass
    class Ranges:

        # Combined head and neck joint gravity relative pitch range induced on head
        pitch: tuple[float, float] = MISSING

        # Gravity relative Head roll range
        roll: tuple[float, float] = MISSING

        # Body relative yaw range
        yaw: tuple[float, float] = MISSING

    # Ranges assigned at HeadPoseCommandCfg construction
    ranges: Ranges = MISSING
