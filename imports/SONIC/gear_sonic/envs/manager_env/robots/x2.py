"""IsaacLab configuration for the AgiBot X2 Ultra with OmniHand."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils


X2_USD_PATH = "gear_sonic/data/robots/x2/x2_ultra_omnihand_isaaclab.usda"

# The order is the active-joint traversal order of the USD-derived IK MJCF. It
# is also used by JointPositionActionCfg so policy actions and reference DOFs
# share one explicit, name-validated contract. The USD contains another 12
# passive finger joints; they intentionally do not appear here.
X2_ACTIVE_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "L_index_abad_joint",
    "L_index_pip_joint",
    "L_middle_pip_joint",
    "L_pinky_abad_joint",
    "L_pinky_pip_joint",
    "L_ring_abad_joint",
    "L_ring_pip_joint",
    "L_thumb_roll_joint",
    "L_thumb_abad_joint",
    "L_thumb_mcp_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "R_index_abad_joint",
    "R_index_pip_joint",
    "R_middle_pip_joint",
    "R_pinky_abad_joint",
    "R_pinky_pip_joint",
    "R_ring_abad_joint",
    "R_ring_pip_joint",
    "R_thumb_roll_joint",
    "R_thumb_abad_joint",
    "R_thumb_mcp_joint",
]

X2_LOWER_BODY_JOINT_NAMES = X2_ACTIVE_JOINT_NAMES[:12]
X2_HAND_JOINT_NAMES = X2_ACTIVE_JOINT_NAMES[24:34] + X2_ACTIVE_JOINT_NAMES[41:51]

_HAND_EFFORT_BY_SUFFIX = {
    "thumb_roll_joint": 0.418,
    "thumb_abad_joint": 0.314,
    "thumb_mcp_joint": 1.764,
    "index_abad_joint": 1.863,
    "index_pip_joint": 1.764,
    "middle_pip_joint": 1.764,
    "ring_abad_joint": 2.766,
    "ring_pip_joint": 1.764,
    "pinky_abad_joint": 3.393,
    "pinky_pip_joint": 1.764,
}
_HAND_STIFFNESS_BY_SUFFIX = {
    "thumb_roll_joint": 1.0,
    "thumb_abad_joint": 1.0,
    "thumb_mcp_joint": 3.0,
    "index_abad_joint": 2.0,
    "index_pip_joint": 3.0,
    "middle_pip_joint": 3.0,
    "ring_abad_joint": 2.0,
    "ring_pip_joint": 3.0,
    "pinky_abad_joint": 2.0,
    "pinky_pip_joint": 3.0,
}


def _hand_values(values_by_suffix: dict[str, float]) -> dict[str, float]:
    return {
        joint_name: value
        for joint_name in X2_HAND_JOINT_NAMES
        for suffix, value in values_by_suffix.items()
        if joint_name.endswith(suffix)
    }

# TrackingCommand resolves both permutations from names after IsaacLab has
# loaded the articulation. This avoids depending on PhysX's internal joint and
# rigid-body traversal order.
X2_MOTION_MAPPING = {
    "resolve_from_mjcf": True,
    "controlled_joint_names": X2_ACTIVE_JOINT_NAMES,
    "lower_joint_names": X2_LOWER_BODY_JOINT_NAMES,
}


X2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=X2_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.64),
        joint_pos={
            ".*_knee_joint": 0.10,
            ".*_ankle_pitch_joint": -0.05,
            "left_shoulder_roll_joint": 0.30,
            "right_shoulder_roll_joint": -0.30,
            ".*_elbow_joint": -0.50,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=120.0,
            stiffness=120.0,
            damping=5.0,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 36.0,
                ".*_ankle_roll_joint": 24.0,
            },
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 30.0,
            },
            damping=2.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
            effort_limit_sim={
                "waist_yaw_joint": 120.0,
                "waist_pitch_joint": 48.0,
                "waist_roll_joint": 48.0,
            },
            stiffness={
                "waist_yaw_joint": 100.0,
                "waist_pitch_joint": 60.0,
                "waist_roll_joint": 60.0,
            },
            damping=4.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim={"head_yaw_joint": 2.6, "head_pitch_joint": 0.6},
            stiffness={"head_yaw_joint": 8.0, "head_pitch_joint": 4.0},
            damping=0.4,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 36.0,
                ".*_shoulder_roll_joint": 36.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
            },
            stiffness=40.0,
            damping=2.0,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_wrist_yaw_joint": 24.0,
                ".*_wrist_pitch_joint": 4.8,
                ".*_wrist_roll_joint": 4.8,
            },
            stiffness={
                ".*_wrist_yaw_joint": 24.0,
                ".*_wrist_pitch_joint": 8.0,
                ".*_wrist_roll_joint": 8.0,
            },
            damping=1.0,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=X2_HAND_JOINT_NAMES,
            effort_limit_sim=_hand_values(_HAND_EFFORT_BY_SUFFIX),
            stiffness=_hand_values(_HAND_STIFFNESS_BY_SUFFIX),
            damping=0.10,
        ),
    },
)


def _build_action_scale() -> dict[str, float]:
    scale = {}
    for actuator in X2_CFG.actuators.values():
        efforts = actuator.effort_limit_sim
        stiffness = actuator.stiffness
        names = actuator.joint_names_expr
        if not isinstance(efforts, dict):
            efforts = dict.fromkeys(names, efforts)
        if not isinstance(stiffness, dict):
            stiffness = dict.fromkeys(names, stiffness)
        for name in names:
            if name in efforts and name in stiffness and stiffness[name]:
                scale[name] = 0.25 * efforts[name] / stiffness[name]
    return scale


X2_ACTION_SCALE = _build_action_scale()
