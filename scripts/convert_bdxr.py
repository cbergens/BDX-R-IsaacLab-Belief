"""Convert the BDX-R URDF file to USDA

`bdxr.py` spawns from the usda file this script generates via UsdFileCfg.

This script is entirely seperate from the bdxr.py file. Ie NVIDIA's shipped converter
isn't used there anymore. It is used here to generate a usda file, that is then sourced
in the task source code.

Output lands in data/Robots/BDXR/converted/URDF/URDF.usda. The importer refuses
to overwrite an existing output dir. Instead, it writes URDF_1/, URDF_2/, ... at
the converted/ dir. Make sure to delete `converted/` if you want the path to stay
stable with a fresh usda.
"""

from pathlib import Path

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import UrdfConverter  # noqa: E402

BDXR_URDF_PATH = Path(__file__).resolve().parents[1] / "source/BDXR/data/Robots/BDXR/URDF.urdf"

BDX_R_URDF_CFG = sim_utils.UrdfConverterCfg(
    asset_path=str(BDXR_URDF_PATH),
    usd_dir=str(BDXR_URDF_PATH.parent / "converted"),
    fix_base=False,
    merge_fixed_joints=True,
    merge_mesh=True,    # Physics is unaffected here, prim count is just reduced for training speed.

    # Gains come from the DelayedPDActuatorCfg groups in bdxr.py.
    # This just initializes everything else to 0.
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    ),
)

print("USD written to:", UrdfConverter(BDX_R_URDF_CFG).usd_path)
app.close()
