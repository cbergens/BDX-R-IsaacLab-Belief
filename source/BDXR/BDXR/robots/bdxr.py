# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Disney BD-X (BDX-R) robot.

* :obj:`BDX_R_CFG`: BDX-R with delayed PD actuators on the legs.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

BDXR_URDF_PATH = Path(__file__).resolve().parents[2] / "data/Robots/BDXR/URDF.urdf"


# Taken from:
# https://github.com/BDX-R/BDX-R-MjLab/blob/main/src/bdx_r_mjlab/robots/bdxr/bdxr_constants.py#L52
class MotorConstants:

    # RobStride 02 Motor constants
    class RobStride02:
        armature = 0.0142
        kp = 16.581
        kd = 1.056
        max_vel = 37.699
        max_eff = 10.9

    # RobStride 03 Motor constants
    class RobStride03:
        armature = 0.06
        kp = 78.957
        kd = 5.027
        max_vel = 18.849
        max_eff = 42.0

    # RobStride 05 Motor constants (formuliac, awaiting chirp testing)
    class RobStride05:
        _damping_ratio = 0.8
        # The head hanging off Head_Pitch is 0.0229 kg.m2
        _load_inertia = 0.0229

        # Cap kp by torque rather than droop. Anything stiffer saturates the 4.2 N.m
        # limit at the ~5 deg of tracking error the head sees, and it chatters
        _linear_range = 0.148

        armature = 7e-4
        max_vel = 45
        max_eff = 3.85
        kp = max_eff / _linear_range
        kd = 2.0 * _damping_ratio * (_load_inertia * kp) ** 0.5

    # TODO: Set true control loop latency. Currently ~15ms
    min_delay = 3
    max_delay = 3

BDX_R_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,

        # Collapses the per-link visual meshes the 3.0 importer emits separately:
        # 166 -> 100 Xform prims per robot. Viewport step cost at 80 envs drops
        # 95.4 -> 77.2 ms. Physics is untouched -- 15 bodies, 14 DOFs, same mass
        merge_mesh=True,
        asset_path=str(BDXR_URDF_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),

        # Self-collision disabled for step time. The property it protects -- no gaits
        # where the legs pass through each other -- is recovered by terminating on
        # upper-leg contact instead
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),

        # Gains come from the actuator model below, not the importer
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        # Foot pads bottom out 0.2445 m below base_link at the pose below
        pos=(0.0, 0.0, 0.2445),

        # The URDF zero pose is already a stance: thigh raked back 44 deg, shank forward
        # 19 deg, whole-body CoM inside the foot support polygon. No crouch to add
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),

    # Soften the joint limit to prevent hard stops and jerk. Creates opportunity for
    # penalty in last 10% RoM
    soft_joint_pos_limit_factor=0.9,

    actuators={

        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Hip_Yaw", ".*_Hip_Roll", ".*_Hip_Pitch", ".*_Knee", ".*_Ankle"],
            stiffness={
                ".*_Hip_Yaw": MotorConstants.RobStride03.kp,
                ".*_Hip_Roll": MotorConstants.RobStride03.kp,
                ".*_Hip_Pitch": MotorConstants.RobStride03.kp,
                ".*_Knee": MotorConstants.RobStride03.kp,
                ".*_Ankle": MotorConstants.RobStride02.kp,
            },
            damping={
                ".*_Hip_Yaw": MotorConstants.RobStride03.kd,
                ".*_Hip_Roll": MotorConstants.RobStride03.kd,
                ".*_Hip_Pitch": MotorConstants.RobStride03.kd,
                ".*_Knee": MotorConstants.RobStride03.kd,
                ".*_Ankle": MotorConstants.RobStride02.kd,
            },
            armature={
                ".*_Hip_Yaw": MotorConstants.RobStride03.armature,
                ".*_Hip_Roll": MotorConstants.RobStride03.armature,
                ".*_Hip_Pitch": MotorConstants.RobStride03.armature,
                ".*_Knee": MotorConstants.RobStride03.armature,
                ".*_Ankle": MotorConstants.RobStride02.armature,
            },
            effort_limit={
                ".*_Hip_Yaw": MotorConstants.RobStride03.max_eff,
                ".*_Hip_Roll": MotorConstants.RobStride03.max_eff,
                ".*_Hip_Pitch": MotorConstants.RobStride03.max_eff,
                ".*_Knee": MotorConstants.RobStride03.max_eff,
                ".*_Ankle": MotorConstants.RobStride02.max_eff,
            },
            effort_limit_sim={
                ".*_Hip_Yaw": MotorConstants.RobStride03.max_eff,
                ".*_Hip_Roll": MotorConstants.RobStride03.max_eff,
                ".*_Hip_Pitch": MotorConstants.RobStride03.max_eff,
                ".*_Knee": MotorConstants.RobStride03.max_eff,
                ".*_Ankle": MotorConstants.RobStride02.max_eff,
            },
            velocity_limit_sim={
                ".*_Hip_Yaw": MotorConstants.RobStride03.max_vel,
                ".*_Hip_Roll": MotorConstants.RobStride03.max_vel,
                ".*_Hip_Pitch": MotorConstants.RobStride03.max_vel,
                ".*_Knee": MotorConstants.RobStride03.max_vel,
                ".*_Ankle": MotorConstants.RobStride02.max_vel,
            },
            min_delay=MotorConstants.min_delay,
            max_delay=MotorConstants.max_delay,
        ),

        "head": DelayedPDActuatorCfg (
            joint_names_expr=["Head_.*", "Neck_.*"],
            stiffness={
                "Head_.*": MotorConstants.RobStride05.kp,
                "Neck_.*": MotorConstants.RobStride02.kp
            },
            damping={
                "Head_.*": MotorConstants.RobStride05.kd,
                "Neck_.*": MotorConstants.RobStride02.kd
            },
            armature={
                "Head_.*": MotorConstants.RobStride05.armature,
                "Neck_.*": MotorConstants.RobStride02.armature
            },
            effort_limit={
                "Head_.*": MotorConstants.RobStride05.max_eff,
                "Neck_.*": MotorConstants.RobStride02.max_eff,
            },
            effort_limit_sim={
                "Head_.*": MotorConstants.RobStride05.max_eff,
                "Neck_.*": MotorConstants.RobStride02.max_eff,
            },
            velocity_limit_sim={
                "Head_.*": MotorConstants.RobStride05.max_vel,
                "Neck_.*": MotorConstants.RobStride02.max_vel
            },
            min_delay=MotorConstants.min_delay,
            max_delay=MotorConstants.max_delay,

        )
    },
)
"""Configuration for the Disney BDX-R robot with delayed PD actuator model."""
