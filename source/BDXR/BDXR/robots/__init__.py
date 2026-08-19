from pxr import UsdPhysics

import isaaclab.sim.schemas.schemas as _schemas
from isaaclab.sim.utils import safe_set_attribute_on_usd_prim
from isaaclab.sim.utils.stage import get_current_stage


def _activate_contact_sensors_nested(prim_path: str, threshold: float = 0.0, stage=None):
    """Patch: IsaacLab 3.0's activate_contact_sensors stops descending at the first rigid
    body, which was safe under the 2.x flat USD layout. The 3.0 URDF importer nests links
    by kinematic chain, so only base_link ever gets PhysxContactReportAPI and every contact
    sensor below the root resolves to nothing. Remove once fixed upstream.
    """

    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    count = 0
    frontier = [prim]
    while frontier:
        child = frontier.pop(0)
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            applied = child.GetAppliedSchemas()
            if "PhysxRigidBodyAPI" not in applied:
                child.AddAppliedSchema("PhysxRigidBodyAPI")
            safe_set_attribute_on_usd_prim(child, "physxRigidBody:sleepThreshold", 0.0, camel_case=False)
            if "PhysxContactReportAPI" not in applied:
                child.AddAppliedSchema("PhysxContactReportAPI")
            safe_set_attribute_on_usd_prim(child,"physxContactReport:threshold", threshold, camel_case=False)
            count += 1
        # the fix: descend unconditionally, links nest under links
        frontier += child.GetChildren()

    if count == 0:
        raise ValueError(f"No contact sensors added to the prim: '{prim_path}'.")
    return True


_schemas.activate_contact_sensors = _activate_contact_sensors_nested

from .bdxr import BDX_R_CFG # noqa: F401