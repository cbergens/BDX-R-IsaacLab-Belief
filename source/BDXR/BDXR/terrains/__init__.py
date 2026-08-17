# Re-export so consumers write `from BDXR.terrains import BDXR_ROUGH_CFG`, matching the
# `from BDXR.robots import BDX_R_CFG` idiom already used by the task configs
"""Terrain configurations for this extension."""

from .rough import BDXR_ROUGH_CFG  # noqa: F401
