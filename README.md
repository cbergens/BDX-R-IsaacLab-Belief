# BDX-R IsaacLab Teacher-Student Belief-State

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
-   **~~Add Camera and~~ Explore ROS2 Bridge:** Add a camera to the robot's head and [Screencast from 09-07-2026 09:33:27 PM.webm](https://github.com/user-attachments/assets/76e0d5df-2b79-48f3-8b0e-7e3c3e8c5c66)
view it using a ROS2 bridge in-sim. This, in addition to issuing commands through ROS2, will gain confidence in VLA control feasibility.
-   **Begin Exploring Student-Teacher Distillation:** Begin researching how to implement imitation learning between two models in Isaac Lab. Train a privileged teacher model.
-   **Begin prototyping physical components:** 3D-print body parts, buy Jetson Orin Nano, and begin exploring potential implementations for additional sensors to open foundation-model opportunities.

*At this stage, the project is concentrated on the fundamental mechanics of the body's movement. Expressiveness and the integration of a head are future goals to be explored after mastering stable locomotion.*



```bash
python play.py --task=Bdxr-Walk-Play-v0 --num_envs 100 --device cpu

```

## 🙏 Community and Acknowledgements

A special thank you to [KaydenKnapik](https://github.com/KaydenKnapik). His willingness to make this project so accessible to the robotics community is incredible, and it serves as a segue into reinforcement learning and physical AI for myself, and I'm sure, many others.
