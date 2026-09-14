# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Noise models for sensors that Isaac Lab's built-ins do not cover."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

__all__ = ["BitFlipNoiseCfg", "bit_flip_noise"]


def bit_flip_noise(data: torch.Tensor, cfg: BitFlipNoiseCfg) -> torch.Tensor:
    """Randomly invert binary values -- the microswitch analogue of Gaussian noise.

    Additive noise is meaningless on a 0/1 signal: what a switch actually gets wrong is
    bouncing on impact and missing marginal loading, both of which read as a flipped bit.

    Note:
        ``cfg.operation`` is ignored. This replaces values rather than combining with
        them, so "add"/"scale"/"abs" have no meaningful interpretation here.
    """
    return torch.where(torch.rand_like(data) < cfg.p, 1.0 - data, data)


@configclass
class BitFlipNoiseCfg(NoiseCfg):
    """Configuration for per-element bit-flip noise on a binary observation."""

    func = bit_flip_noise

    p: float = 0.02
    """Per-element, per-step probability of inverting the bit.

    Real switch bounce lasts a few ms on impact, which puts the defensible band at 1-3%.
    """
