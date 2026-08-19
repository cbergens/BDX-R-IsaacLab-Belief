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

BDX_R_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,

        # Collapses the per-link visual meshes the 3.0 importer emits separately:
        # 166 -> 100 Xform prims per robot. Viewport step cost at 80 envs drops
        # 95.4 -> 77.2 ms. Physics is untouched -- 11 bodies, 10 DOFs, same mass
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
        # Foot pads bottom out 0.330 m below base_link at the pose below
        pos=(0.0, 0.0, 0.33),

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
                ".*_Hip_Yaw": 78.957,
                ".*_Hip_Roll": 78.957,
                ".*_Hip_Pitch": 78.957,
                ".*_Knee": 78.957,
                ".*_Ankle": 16.581,
            },
            damping={
                ".*_Hip_Yaw": 5.027,
                ".*_Hip_Roll": 5.027,
                ".*_Hip_Pitch": 5.027,
                ".*_Knee": 5.027,
                ".*_Ankle": 1.056,
            },
            armature={
                ".*_Hip_Yaw": 0.02,
                ".*_Hip_Roll": 0.02,
                ".*_Hip_Pitch": 0.02,
                ".*_Knee": 0.02,
                ".*_Ankle": 0.0042,
            },
            effort_limit_sim={
                ".*_Hip_Yaw": 42.0,
                ".*_Hip_Roll": 42.0,
                ".*_Hip_Pitch": 42.0,
                ".*_Knee": 42.0,
                ".*_Ankle": 11.9,
            },
            velocity_limit_sim={
                ".*_Hip_Yaw": 18.849,
                ".*_Hip_Roll": 18.849,
                ".*_Hip_Pitch": 18.849,
                ".*_Knee": 18.849,
                ".*_Ankle": 37.699,
            },
            min_delay=0,
            max_delay=0,
        ),
    },
)
"""Configuration for the Disney BD-X robot with delayed PD actuator model."""
