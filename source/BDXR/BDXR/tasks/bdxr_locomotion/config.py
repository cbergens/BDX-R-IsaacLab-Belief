# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tunables for the BDX-R walk task, grouped so every number has one home.

Term definitions in bdxr_env_cfg.py read from the singletons at the bottom of this
file rather than carrying literals, so a retune touches exactly one place.
"""

from isaaclab.utils import configclass

# Every constant here is derived from the Open Duck Mini v2 by Froude similarity. The
# length scale is CoM height above the contact plane at the nominal stance -- the
# inverted-pendulum length -- measured by FK on both URDFs: duck 0.2161 m, BDX-R
# 0.3029 m, so L = 1.402 and sqrt(L) = 1.184. Lengths scale by L, times and linear
# velocities by sqrt(L), angular velocities by its inverse


@configclass
class JointCfg:
    """Actuated joint names, explicit and ordered.

    Pair with preserve_order=True at every call site: the exported policy indexes the
    observation and action vectors positionally, so this order is part of its contract.
    """

    legs: list[str] = [
        "Left_Hip_Yaw",   "Right_Hip_Yaw",
        "Left_Hip_Roll",  "Right_Hip_Roll",
        "Left_Hip_Pitch", "Right_Hip_Pitch",
        "Left_Knee",      "Right_Knee",
        "Left_Ankle",     "Right_Ankle",
    ]

    # Held at default by their own actuator group in bdxr.py; not in the action space
    head: list[str] = ["Neck_Pitch", "Head_Pitch", "Head_Yaw", "Head_Roll"]


@configclass
class BodyCfg:
    """Link-name patterns. Case matters -- the URDF exporter capitalised every link."""

    base: str = "base_link"
    feet: str = ".*_Foot"

    # The thigh survives merge_fixed_joints under its motor's name; Upper_Leg is folded
    # into it. Terminating here catches the leg-crossing gaits that self-collisions would
    illegal_contact: list[str] = ["base_link", ".*_Hip_Pitch_Motor"]


@configclass
class GaitCfg:
    """Placo walk schedule, Froude-scaled from the duck's identified gait.

    Re-derive against the BDX-R URDF before a long run; the scaling is a sanity check
    on Placo's output, not a replacement for it.
    """

    # 2*single_support(0.2017) + 2*double_support(0.0363), the duck's 0.170/0.0306
    # primitives scaled by sqrt(1.408)
    period: float = 0.476
    """Full gait cycle, seconds."""

    # Stance fraction per foot -> 15% double support. Dimensionless, so it is invariant
    # under the scale change
    duty: float = 0.577
    """Stance fraction of one foot's cycle."""

    swing_height: float = 0.056
    """Peak swing clearance, metres. Scales with the length ratio."""

    swing_std: float = 0.056
    """Gaussian width on the swing-height error. A length tolerance, so it scales too."""

    ref_speed: float = 0.297
    """Command speed at which swing amplitude reaches swing_height."""

    cmd_threshold: float = 0.05
    """Below this commanded speed the gait terms gate off, or they pay for marching in place."""

    air_time_threshold: float = 0.202
    """Air time the feet are rewarded up to; equals the scaled single-support duration."""


@configclass
class ActionCfg:
    """Joint-position action shaping."""

    scale: float = 0.25
    """Action-to-radian gain. Joint angles are dimensionless, so this does not scale."""

    # Slowest actuated joint (the hips and knees at 18.849 rad/s); using the ankle's
    # faster limit would let the limiter command hip targets physics cannot follow
    max_joint_velocity: float = 18.849
    """Rate ceiling on the position target, rad/s."""


@configclass
class CommandCfg:
    """Velocity command sampling, Froude-scaled from the duck's trained envelope."""

    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    rel_standing_envs: float = 0.2
    rel_heading_envs: float = 1.0
    heading_control_stiffness: float = 0.5

    lin_vel_x: tuple[float, float] = (-0.297, 0.451)
    lin_vel_y: tuple[float, float] = (-0.178, 0.178)
    ang_vel_z: tuple[float, float] = (-0.843, 0.843)


@configclass
class RewardWeightCfg:
    """Reward weights. Duck values except where the scale change moves the term's magnitude."""

    track_lin_vel_xy: float = 1.0
    track_ang_vel_z: float = 1.0
    feet_air_time: float = 0.5
    gait_contact: float = 1.0
    foot_swing_height: float = 1.0

    # Penalises squared sag, so the magnitude grows with the length ratio squared
    min_height: float = -5.0

    feet_slide: float = -0.1
    z_vel: float = -0.05
    ang_vel_xy: float = -0.05
    flat_orientation: float = -0.2
    joint_pos_limits: float = -3.0

    # Applied torque scales with m*g*L
    joint_torques: float = -1.2e-5

    # Joint acceleration scales with the inverse length ratio, squared by the term
    joint_acc: float = -1.25e-7

    action_rate: float = -0.01
    stand_still: float = -0.5
    termination: float = -50.0


@configclass
class RewardParamCfg:
    """Non-weight reward parameters."""

    # Root to lowest foot body at the URDF zero pose is 0.1636 m, and the duck sets its
    # floor at 98.9% of the same measurement
    min_base_height: float = 0.162
    """Trunk height above the lower foot below which the sag penalty engages."""

    track_lin_vel_std: float = 0.2
    track_ang_vel_std: float = 0.5


@configclass
class ObsNoiseCfg:
    """Sensor noise magnitudes, matched to the hardware they stand in for."""

    imu_ang_vel: float = 0.05
    imu_gravity: float = 0.05

    # Mean IMU gravity noise ~2.3 deg plus the pitch-accuracy term
    imu_gravity_bias: float = 0.04

    joint_pos: float = 0.01
    joint_vel: float = 1.5

    # Per-foot, per-step probability of an inverted contact bit; ~1 spurious flip per
    # foot per second at 50 Hz
    foot_contact_flip: float = 0.02


@configclass
class EventRangeCfg:
    """Randomisation ranges. Duck values; masses are absolute and scale with the robot."""

    static_friction: tuple[float, float] = (0.45, 1.4)
    dynamic_friction: tuple[float, float] = (0.5, 1.2)
    restitution: tuple[float, float] = (0.0, 0.1)

    # The duck's -0.05/+0.20 kg on a 2.06 kg robot, held at the same fraction of mass
    base_mass: tuple[float, float] = (-0.465, 1.862)

    # +-20% on the servos, for wear and unit-to-unit spread
    actuator_gain: tuple[float, float] = (0.8, 1.2)

    push_interval_s: tuple[float, float] = (3.0, 7.0)
    push_lin_vel: tuple[float, float] = (-1, 1)
    push_lin_z_vel: tuple[float, float] = (-0.25, 0.25)
    push_ang_vel: tuple[float, float] = (-0.7, 0.7)

    reset_joint_scale: tuple[float, float] = (0.95, 1.05)


# Singletons read by bdxr_env_cfg.py
JOINTS = JointCfg()
BODIES = BodyCfg()
GAIT = GaitCfg()
ACTION = ActionCfg()
COMMANDS = CommandCfg()
WEIGHTS = RewardWeightCfg()
REWARD_PARAMS = RewardParamCfg()
OBS_NOISE = ObsNoiseCfg()
EVENTS = EventRangeCfg()
