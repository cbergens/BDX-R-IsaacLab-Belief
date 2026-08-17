# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain configurations tuned for BDX-R.

* :obj:`BDXR_ROUGH_CFG`: Mixed flat/rough/sloped terrain at BDX-R scale.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

# Every length here is the Open Duck Mini v2 terrain scaled by 1.408, the ratio of the
# two robots' pendulum lengths. Slopes are dimensionless and carry over unchanged
BDXR_ROUGH_CFG = TerrainGeneratorCfg(
    size=(5.6, 5.6),
    border_width=14.0,
    num_rows=8,
    num_cols=10,
    horizontal_scale=0.07,
    vertical_scale=0.003,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(0.007, 0.035), noise_step=0.007, border_width=0.35
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.2), platform_width=1.4, border_width=0.35
        ),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.2), platform_width=1.4, border_width=0.35
        ),
        "stair_step": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.014, 0.056), step_width=0.56, platform_width=1.4, border_width=0.35
        ),
    },
)
"""Rough terrain configuration scaled for BDX-R."""
