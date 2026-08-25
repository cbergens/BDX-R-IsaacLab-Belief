# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, PvaCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.physics import PhysxCfg

from . import mdp
from .config import ACTION, BODIES, COMMANDS, EVENTS, GAIT, JOINTS, OBS_NOISE, REWARD_PARAMS, WEIGHTS

##
# Pre-defined configs
##

from BDXR.robots import BDX_R_CFG  # isort: skip
from BDXR.terrains import BDXR_ROUGH_CFG  # isort: skip


##
# Scene definition
##


@configclass
class BdxrSceneCfg(InteractiveSceneCfg):
    """Configuration for a BDX-R scene."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=BDXR_ROUGH_CFG,
        # None spreads spawns over every difficulty row; 0 pinned all envs to row 0
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = BDX_R_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        offset=ImuCfg.OffsetCfg(
            # The URDF's imu link, forward-kinematicked into base_link. The fixed joints
            # along the way are all identity in rotation, so the frames stay aligned
            pos=(-0.02397, 0.02095, 0.17390),
            rot=(0.0, 0.0, 0.0, 1.0),  # (x, y, z, w)
        ),
    )

    pva = PvaCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        offset=PvaCfg.OffsetCfg(
            pos=(-0.02397, 0.02095, 0.17390),
            rot=(0.0, 0.0, 0.0, 1.0),  # (x, y, z, w)
        )
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Joint position action, limit the rate at which the action can change
    joint_pos = mdp.RateLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINTS.legs,
        preserve_order=True,
        scale=ACTION.scale,
        use_default_offset=True,
        max_joint_velocity=ACTION.max_joint_velocity,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # Policy (Actor) Cfg
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Gyro + IMU
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            noise=Unoise(n_min=-OBS_NOISE.imu_ang_vel, n_max=OBS_NOISE.imu_ang_vel),
        )

        pva_projected_gravity = ObsTerm(
            func=mdp.pva_projected_gravity,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-OBS_NOISE.imu_gravity, n_max=OBS_NOISE.imu_gravity),
                bias_noise_cfg=Unoise(n_min=-OBS_NOISE.imu_gravity_bias, n_max=OBS_NOISE.imu_gravity_bias),
            ),
        )

        # Leg joint positions
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS.legs, preserve_order=True)},
            noise=Unoise(n_min=-OBS_NOISE.joint_pos, n_max=OBS_NOISE.joint_pos),
        )

        # Joint velocity
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS.legs, preserve_order=True)},
            noise=Unoise(n_min=-OBS_NOISE.joint_vel, n_max=OBS_NOISE.joint_vel),
        )

        # Contact Sensors
        # Bit-flip rather than additive noise: the switch's failure mode is a wrong bit
        # (bounce on impact, missed marginal loading), not drift. Routed through `noise=`
        # so CriticCfg's enable_corruption=False strips it
        feet_contact = ObsTerm(
            func=mdp.feet_contact_binary,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BODIES.feet)},
            noise=mdp.BitFlipNoiseCfg(p=OBS_NOISE.foot_contact_flip),
        )

        # Command observations
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # Gait phase clock observation
        gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={
                "command_name": "base_velocity",
                "period": GAIT.period,
                "cmd_threshold": GAIT.cmd_threshold
            }
        )

        base_mass = ObsTerm(
            func=mdp.body_mass,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=BODIES.base)},
        )

        # Action observations
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    # Critic Cfg
    @configclass
    class CriticCfg(PolicyCfg):
        """Critic inherits everything actor sees + sim ground truth"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        foot_friction = ObsTerm(func=mdp.foot_friction)

        feet_contact_force = ObsTerm(
            func=mdp.feet_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BODIES.feet)},
        )

        def __post_init__(self) -> None:
            super().__post_init__()
            self.enable_corruption = False  # This data is ground truth - no corruption

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CommandsCfg:
    """Configuration for velocity commands"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=COMMANDS.resampling_time_range,
        rel_standing_envs=COMMANDS.rel_standing_envs,
        rel_heading_envs=COMMANDS.rel_heading_envs,
        heading_command=True,
        heading_control_stiffness=COMMANDS.heading_control_stiffness,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=COMMANDS.lin_vel_x,
            lin_vel_y=COMMANDS.lin_vel_y,
            ang_vel_z=COMMANDS.ang_vel_z,
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # Randomizes physical properties of the robot's rigid bodies, creating the
    # emulation of walking on different surfaces
    physics_props = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": EVENTS.static_friction,
            "dynamic_friction_range": EVENTS.dynamic_friction,
            "restitution_range": EVENTS.restitution,
            "num_buckets": 64,
        },
    )

    # Base mass variance for battery, compute, print variance and payload capacity
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BODIES.base),
            "mass_distribution_params": EVENTS.base_mass,
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    # Randomize actuator gains at startup to emulate different servos and
    # wear-and-tear for sim2real
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS.legs),
            "stiffness_distribution_params": EVENTS.actuator_gain,
            "damping_distribution_params": EVENTS.actuator_gain,
            "operation": "scale",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={  # Randomize spawn position for generalization
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        # Small insignificant starting pose variance
        params={"position_range": EVENTS.reset_joint_scale, "velocity_range": (0.0, 0.0)},
    )

    # Interval - apply a pushing velocity change to test recovery
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=EVENTS.push_interval_s,
        params={
            "velocity_range": {
                "x": EVENTS.push_lin_vel,
                "y": EVENTS.push_lin_vel,
                "z": EVENTS.push_lin_z_vel,
                "roll": EVENTS.push_ang_vel,
                "pitch": EVENTS.push_ang_vel,
            }
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Velocity tracking reward
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=WEIGHTS.track_lin_vel_xy,
        params={"command_name": "base_velocity", "std": REWARD_PARAMS.track_lin_vel_std},
    )

    # Angular velocity tracking reward
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=WEIGHTS.track_ang_vel_z,
        params={"command_name": "base_velocity", "std": REWARD_PARAMS.track_ang_vel_std},
    )

    # Reward the robot for keeping one foot off the ground while moving for
    # "threshold" amount of time
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=WEIGHTS.feet_air_time,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BODIES.feet),
            "threshold": GAIT.air_time_threshold,
        },
    )

    # Minimum height penalty - penalize the robot for sagging too low to the ground
    min_height = RewTerm(
        func=mdp.base_height_floor_l2,
        weight=WEIGHTS.min_height,
        params={
            "minimum_height": REWARD_PARAMS.min_base_height,
            "asset_cfg": SceneEntityCfg("robot", body_names=BODIES.feet),
        },
    )

    # Prevents the single foot skating that feet_air_time otherwise pays for
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=WEIGHTS.feet_slide,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BODIES.feet),
            "asset_cfg": SceneEntityCfg("robot", body_names=BODIES.feet),
        },
    )

    # Gait rhythm reward, scores foot contact against alternating pace schedule
    gait_contact = RewTerm(
        func=mdp.gait_contact_schedule,
        weight=WEIGHTS.gait_contact,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BODIES.feet),
            "period": GAIT.period,
            "duty": GAIT.duty,
            "cmd_threshold": GAIT.cmd_threshold,
        },
    )

    # Swing shape tracking reward
    foot_swing_height = RewTerm(
        func=mdp.foot_swing_height_track,
        weight=WEIGHTS.foot_swing_height,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=BODIES.feet),
            "period": GAIT.period,
            "duty": GAIT.duty,
            "swing_height": GAIT.swing_height,
            "ref_speed": GAIT.ref_speed,
            "std": GAIT.swing_std,
            "cmd_threshold": GAIT.cmd_threshold,
        },
    )

    # Z Velocity Penalty - penalize the robot for moving up or down too quickly
    z_vel = RewTerm(func=mdp.lin_vel_z_l2, weight=WEIGHTS.z_vel)

    # Pitch Roll velocity Penalty
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=WEIGHTS.ang_vel_xy)

    # Flat orientation (pitch roll position) penalty
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=WEIGHTS.flat_orientation)

    # Penalize joints in the last 10% of their RoM
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=WEIGHTS.joint_pos_limits,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS.legs)},
    )

    # Joint Torque Penalty - penalize the robot for straining servos in joints
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=WEIGHTS.joint_torques,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS.legs)},
    )

    # Joint Acceleration Penalty - penalize the robot for straining servos in joints
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=WEIGHTS.joint_acc,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS.legs)},
    )

    # Action Rate Penalty - keeps policy from changing motor commands too abruptly
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=WEIGHTS.action_rate)

    # Penalizes the robot for moving when command is ~0 magnitude
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=WEIGHTS.stand_still,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS.legs),
        },
    )

    # Termination Penalty - penalizes the robot for death
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=WEIGHTS.termination)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Fall over terms
    fell_over = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BODIES.illegal_contact),
            "threshold": 1.0,
        },
        time_out=False,
    )


##
# Environment configurations
##


@configclass
class BdxrEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BdxrSceneCfg = BdxrSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # Simulation and physics settings
    sim: SimulationCfg = SimulationCfg(
        physics=PhysxCfg(
            gpu_collision_stack_size=2**28,  #256 MB
            gpu_max_rigid_patch_count=10 * 2**15,
        )
    )

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        # 1/200 with decimation 4 gives a 50 Hz policy, the rate the hardware control
        # loop targets. action_rate_l2 is tuned against that step size, so changing it
        # after training invalidates the policy
        self.decimation = 4
        self.sim.dt = 1 / 200
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.render_interval = self.decimation
        # Make sure sim and contact forces have matching time deltas
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class BdxrEnvCfg_PLAY(BdxrEnvCfg):
    """Small, cheap variant for visual inspection. Physics identical -- only scene size differs."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 80
        self.scene.env_spacing = 3.0

        # Deep copy so shrinking the play grid does not mutate the shared training cfg
        tg = copy.deepcopy(self.scene.terrain.terrain_generator)
        tg.num_rows = 3
        tg.num_cols = 16
        self.scene.terrain.terrain_generator = tg

        # Observation noise exists for training robustness; it only makes visual
        # debugging harder to interpret
        self.observations.policy.enable_corruption = False
