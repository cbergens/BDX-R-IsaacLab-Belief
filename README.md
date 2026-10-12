# BDX-R IsaacLab Teacher-Student Belief-State

![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-22314E?logo=ros&logoColor=white)
![Isaac Sim 6.1.0](https://img.shields.io/badge/Isaac%20Sim-6.1.0-76B900?logo=nvidia&logoColor=white)
![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-TorchScript-EE4C2C?logo=pytorch&logoColor=white)
![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)

This repository has been forked from [KaydenKnapik](https://github.com/KaydenKnapik)'s [BDX-R-IsaacLab repository](https://github.com/BDX-R/BDX-R-IsaacLab). Huge thanks to him for the help he's offered throughout this project.

## Overview

This repository covers the development of BDX-R, a personal/experimental endeavor to create a teacher-student distilled bipedal robot inspired by Disney's BDX droids. The primary goal is to validate teacher-student distillation in bipedal policies and implement a belief state, as cited in ETH Zurich and Intel's research paper, [here](https://arxiv.org/pdf/2201.08117).

## Some images from Kayden's README:

| BDX-R in Isaac Lab Simulation | BDX-R Physical Prototype|
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/7b92c5b6-71ba-4746-a2d3-77d880e18014" width="500" /> | <img src="https://github.com/user-attachments/assets/4f65d9e9-85ad-497f-b687-10c54377d0f2" width="302" /> |

---

## Current Policy:

_BDX-R moving down stairs with a disturbance:_

[BDX-R Rough Terrain with Disturbance](https://github.com/user-attachments/assets/28d0ae42-52a3-4c8b-8a4b-85e8e2bb3e92)

## 🎯 Current Focus: Walking and Sim2Real

Currently, a sim-robust policy exists; however, our next objectives are the following:

-   ~~**Update URDF kinematics:** Pull the more up-to-date version of the robot from Kayden's mujoco repository, and reconfigure it in Isaac Lab.~~
-   ~~**Unlock Head and Neck Joints:** This could be a result of the above objective; however, currently, the robot's head and neck joints are static.~~
-   ~~**Add Camera and Explore ROS2 Bridge:** Add a camera to the robot's head and [Screencast from 09-07-2026 09:33:27 PM.webm](https://github.com/user-attachments/assets/76e0d5df-2b79-48f3-8b0e-7e3c3e8c5c66)~~
~~view it using a ROS2 bridge in-sim. This, in addition to issuing commands through ROS2~~ will gain confidence in VLA control feasibility.
	- ~~Study [this humanoid control node example](https://github.com/isaac-sim/IsaacSim-ros_workspaces/tree/main/humble_ws/src/humanoid_locomotion_policy_example/h1_fullbody_controller) to begin construction of a BDXR ROS2 control node for in-sim policy validation.~~ 
	- Controller repository [here](https://github.com/cbergens/BDXR_controller)
	- Notes from sim control prototyping:
		- Feet need to lift up while turning in place -> edit gait clock command cutoff to also consider base angular velocity commands.
		- Potentially slow down gait clock overall. The robot walks quite fast currently.
-   **Begin Exploring Student-Teacher Distillation:** Begin researching how to implement imitation learning between two models in Isaac Lab. Train a privileged teacher model.
-   **Begin prototyping physical components:** 3D-print body parts, buy Jetson Orin Nano, and begin exploring potential implementations for additional sensors to open foundation-model opportunities.

*At this stage, the project is concentrated on the fundamental mechanics of the body's movement. Expressiveness and the integration of a head are future goals to be explored after mastering stable locomotion.*



```bash
python play.py --task=Bdxr-Walk-Play-v0 --num_envs 100 --device cpu

```

## 📝 Notes: Sim2Real Considerations

-   **Head roll firmware limit:** Limit `Head_Roll` to ±0.25 rad in the motor firmware. Past that, with `Head_Yaw` near -0.6 rad, `Head_Pitch_Motor_Support` touches `Head_Base_Plate` on the real parts. The URDF and the USD already use ±0.25 rad.
-   **Firmware limits match the URDF:** Set every joint's firmware limit to its URDF limit. Self-collision does not guard the filtered pairs below, so the joint limits are the only guard.
-   **Filtered collision pairs:** Self-collision is on, but four body pairs skip collision: `base_link` with each `Hip_Roll_Connector`, each `Hip_Top_Holder` with its `Hip_Pitch_Motor`, and `Head_Pitch_Motor_Support` with `Head_Roll_Motor`. Their convex hulls overlap, but the real parts keep at least 3 mm of clearance across the joint limits. If you widen a joint limit, sweep these pairs again.
-   **Re-converting the robot:** `scripts/convert_bdxr.py` writes the filtered pairs into the generated physics layer. Re-run it after any URDF change, instead of editing `converted/` by hand.
-   **Hull size:** The leg hulls are slightly larger than the real parts, so the policy keeps its legs a little farther apart than it must.
-   **Control latency:** Training delays each joint command by 15 ms (3 physics steps of `DelayedPDActuator`). Measure the real control loop latency and update `min_delay` and `max_delay` in `bdxr.py` to match.

## 🙏 Community and Acknowledgements

A special thank you to [KaydenKnapik](https://github.com/KaydenKnapik). His willingness to make this project so accessible to the robotics community is incredible, and it serves as a segue into reinforcement learning and physical AI for myself, and I'm sure, many others.
