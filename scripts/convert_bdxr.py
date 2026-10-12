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
from pxr import Usd, UsdPhysics  # noqa: E402

BDXR_URDF_PATH = Path(__file__).resolve().parents[1] / "src/BDXR/data/Robots/BDXR/URDF.urdf"

# ponytail: hulls overlap where real parts clear, so these skip collision. Decompose them if one must collide.
FILTERED_PAIRS = {
    "/URDF/Geometry/base_link": [
        "/URDF/Geometry/base_link/Left_Hip_Top_Holder/Left_Hip_Roll_Connector",
        "/URDF/Geometry/base_link/Right_Hip_Top_Holder/Right_Hip_Roll_Connector",
    ],
    "/URDF/Geometry/base_link/Neck/Head_Pitch_Motor_Support": [
        "/URDF/Geometry/base_link/Neck/Head_Pitch_Motor_Support/Head_Roll_Connector/Head_Roll_Motor",
    ],
    "/URDF/Geometry/base_link/Left_Hip_Top_Holder": [
        "/URDF/Geometry/base_link/Left_Hip_Top_Holder/Left_Hip_Roll_Connector/Left_Hip_Pitch_Motor",
    ],
    "/URDF/Geometry/base_link/Right_Hip_Top_Holder": [
        "/URDF/Geometry/base_link/Right_Hip_Top_Holder/Right_Hip_Roll_Connector/Right_Hip_Pitch_Motor",
    ],
}

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

usd_path = UrdfConverter(BDX_R_URDF_CFG).usd_path

# The URDF has no filtered-pair tag, so author the pairs into the generated physics layer.
physics = Usd.Stage.Open(str(Path(usd_path).parent / "payloads/Physics/physics.usda"))
for body, others in FILTERED_PAIRS.items():
    UsdPhysics.FilteredPairsAPI.Apply(physics.GetPrimAtPath(body)).CreateFilteredPairsRel().SetTargets(others)
physics.GetRootLayer().Save()

print("USD written to:", usd_path)
app.close()
