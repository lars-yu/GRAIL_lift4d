"""Adapter for the public ``general_motion_retargeting`` (GMR) package.

GRAIL's retargeting pipeline diverges from public GMR in three places. Rather
than editing files inside the submodule (which forces ``imports/GMR`` to a
"dirty" working tree and complicates submodule bumps), we monkey-patch the
public package at import time. Importing this module is enough to activate
the patches; downstream code does:

    from grail.adapters.gmr import GMR, ROBOT_XML_DICT, ...

and gets the patched GMR transparently.

Patches applied:

1. ``GeneralMotionRetargeting.__init__`` — public GMR multiplies
   ``ik_config["human_scale_table"][k]`` by a per-instance ``ratio``. GRAIL's
   SMPL-X inputs already provide the correct ratio embedded in the SMPL-X
   betas, so we want identity (1.0). After the public init runs, every value
   in ``self.human_scale_table`` is reset to 1.0.

2. ``general_motion_retargeting.utils.smpl.load_smplx_file`` — public GMR
   only knows how to load ``.npz`` SMPL-X dumps. GRAIL's
   ``grail.pipelines.recon_4dhoi`` writes ``.pkl`` files with SMPL-X data wrapped
   under a ``human_data`` key. The replacement function transparently
   handles both ``.pkl`` (GRAIL) and ``.npz`` (public) formats; it also
   (a) truncates ``betas`` to the first 10 dims (public GMR keeps all 16),
   (b) zeroes out root translation before body-model evaluation. Public
   ``.npz`` callers see no change.

3. ``GeneralMotionRetargeting.update_targets`` — public GMR applies only the
   table-1 offsets to every scaled human key. That fails when table 2 adds
   X2-only fingertip landmarks and also leaves table-2 offsets unused. The
   replacement builds each task table from its own key set. For keys shared
   by both tables it intentionally keeps the historical table-1 offsets so
   existing G1 trajectories do not change.
"""

from __future__ import annotations

import logging
import pickle

try:
    import general_motion_retargeting as _gmr
    import general_motion_retargeting.utils.smpl as _gmr_smpl
except ModuleNotFoundError as exc:
    if exc.name != "general_motion_retargeting":
        raise
    raise ModuleNotFoundError(
        "GMR is required for retargeting but is not installed. From the GRAIL "
        "repository root, run `git submodule update --init imports/GMR` and "
        "`python -m pip install -e imports/GMR` in the active environment."
    ) from exc
import mink
import numpy as np
import torch

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch 1: human_scale_table -> identity (override public per-joint ratio).
# ---------------------------------------------------------------------------
_orig_gmr_init = _gmr.GeneralMotionRetargeting.__init__


def _patched_gmr_init(self, *args, **kwargs):
    # Public GMR builds velocity limits from actuator names and assumes they
    # equal joint names. X2 uses motor_<joint> actuator names, so build the
    # limit table from each actuator's actual transmission joint instead.
    use_velocity_limit = kwargs.pop("use_velocity_limit", False)
    _orig_gmr_init(self, *args, use_velocity_limit=False, **kwargs)
    if use_velocity_limit:
        velocity_limits = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_name = self.model.joint(joint_id).name
            velocity_limits[joint_name] = 3 * np.pi
        self.ik_limits.append(mink.VelocityLimit(self.model, velocity_limits))
    if hasattr(self, "human_scale_table"):
        for key in list(self.human_scale_table.keys()):
            self.human_scale_table[key] = 1.0

_gmr.GeneralMotionRetargeting.__init__ = _patched_gmr_init


# ---------------------------------------------------------------------------
# Patch 2: apply task offsets only to keys used by each IK table.
# ---------------------------------------------------------------------------
def _task_target_data(self, human_data, body_names, *, prefer_table1):
    missing = sorted(set(body_names) - set(human_data))
    if missing:
        raise KeyError(f"Missing human IK targets: {missing}")

    pos_offsets = {}
    rot_offsets = {}
    for body_name in body_names:
        if prefer_table1 and body_name in self.pos_offsets1:
            pos_offsets[body_name] = self.pos_offsets1[body_name]
            rot_offsets[body_name] = self.rot_offsets1[body_name]
        else:
            pos_offsets[body_name] = self.pos_offsets2[body_name]
            rot_offsets[body_name] = self.rot_offsets2[body_name]

    subset = {body_name: human_data[body_name] for body_name in body_names}
    return self.offset_human_data(subset, pos_offsets, rot_offsets)


def _patched_update_targets(self, human_data, offset_to_ground=False):
    human_data = self.to_numpy(human_data)
    human_data = self.scale_human_data(
        human_data, self.human_root_name, self.human_scale_table
    )
    # Public GMR's scale_human_data returns tuple values, while
    # apply_ground_offset mutates the position slot in place.
    human_data = {
        body_name: [position, rotation]
        for body_name, (position, rotation) in human_data.items()
    }
    human_data = self.apply_ground_offset(human_data)

    table1_names = tuple(self.human_body_to_task1)
    table2_names = tuple(self.human_body_to_task2)
    table1_data = _task_target_data(
        self, human_data, table1_names, prefer_table1=True
    )
    table2_data = _task_target_data(
        self,
        human_data,
        table2_names,
        # Preserve the public GMR behavior for keys shared with table 1.
        prefer_table1=True,
    )

    if offset_to_ground:
        grounded_table1 = self.offset_human_data_to_ground(table1_data)
        root_shift = (
            grounded_table1[self.human_root_name][0]
            - table1_data[self.human_root_name][0]
        )
        table1_data = grounded_table1
        for body_name, (pos, quat) in table2_data.items():
            table2_data[body_name] = [pos + root_shift, quat]

    self.scaled_human_data = {**table1_data, **table2_data}

    if self.use_ik_match_table1:
        for body_name, task in self.human_body_to_task1.items():
            pos, rot = table1_data[body_name]
            task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
            )

    if self.use_ik_match_table2:
        for body_name, task in self.human_body_to_task2.items():
            pos, rot = table2_data[body_name]
            task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
            )


_gmr.GeneralMotionRetargeting.update_targets = _patched_update_targets


# ---------------------------------------------------------------------------
# Patch 3: smpl.load_smplx_file — accept GRAIL .pkl + truncated betas.
# ---------------------------------------------------------------------------
import smplx as _smplx  # noqa: E402  (imported here so the patch is self-contained)


def _grail_load_smplx_file(smplx_file, smplx_body_model_path):
    """Drop-in replacement that supports GRAIL ``.pkl`` SMPL-X dumps.

    For ``.npz`` inputs the behavior matches public GMR except that we use
    the first 10 betas (public uses all 16) and zero out root translation
    before body-model forward — both of which GRAIL relies on.
    """
    if smplx_file.endswith(".pkl"):
        with open(smplx_file, "rb") as f:
            pkl_data = pickle.load(f)["human_data"]
        gender = "neutral"
        smplx_data = {
            "pose_body": pkl_data["poses"][..., 3:66],
            "root_orient": pkl_data["poses"][..., :3],
            "betas": pkl_data["betas"],
            "trans": pkl_data["trans"],
            "mocap_frame_rate": torch.tensor(30),
        }
        scale = torch.tensor(pkl_data.get("scale", 1.0))
    else:
        smplx_data = np.load(smplx_file, allow_pickle=True)
        gender = str(smplx_data["gender"])
        scale = torch.tensor(1.0)

    body_model = _smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=gender,
        use_pca=False,
    )
    num_frames = smplx_data["pose_body"].shape[0]
    transl = (
        smplx_data["trans"].copy()
        if hasattr(smplx_data["trans"], "copy")
        else np.array(smplx_data["trans"]).copy()
    )
    transl[..., :3] = 0.0

    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"][..., :10]).float().view(1, -1),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(transl).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float() * 10,
        right_hand_pose=torch.zeros(num_frames, 45).float() * 10,
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )
    smplx_output.vertices *= scale
    smplx_output.joints *= scale

    if len(smplx_data["betas"].shape) == 1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]

    return smplx_data, body_model, smplx_output, human_height


_gmr_smpl.load_smplx_file = _grail_load_smplx_file

_logger.debug("Applied GRAIL GMR runtime patches (scale=1.0 + .pkl SMPL-X loader).")


# ---------------------------------------------------------------------------
# Public re-exports — callers do `from grail.adapters.gmr import GMR, ...`.
# ---------------------------------------------------------------------------
from general_motion_retargeting import (  # noqa: E402
    IK_CONFIG_DICT,
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
    GeneralMotionRetargeting as GMR,
    RobotMotionViewer,
)
from general_motion_retargeting.robot_motion_viewer import draw_frame  # noqa: E402

__all__ = [
    "GMR",
    "IK_CONFIG_DICT",
    "ROBOT_BASE_DICT",
    "ROBOT_XML_DICT",
    "VIEWER_CAM_DISTANCE_DICT",
    "RobotMotionViewer",
    "draw_frame",
]
