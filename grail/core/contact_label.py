import json
import os
import re
import sys
from collections.abc import Sequence
from typing import List, Optional

import numpy as np

from grail.adapters.openai_api import DEFAULT_REASONING_MODEL, chat_with_image

# Allowed joints using canonical SMPL-style names (L_/R_)
ALLOWED_CONTACT_JOINTS: list[str] = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    # Fingers merged into these two categories
    "L_Hand",
    "R_Hand",
]


_FINGER_KEYWORDS = ("Index", "Middle", "Pinky", "Ring", "Thumb", "Finger")


def _unique_preserve_order(items: Sequence[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            result.append(it)
    return result


def _token_to_allowed(token: str) -> str | None:
    t = token.strip()
    if not t:
        return None

    # Direct match to allowed
    if t in ALLOWED_CONTACT_JOINTS:
        return t

    # Normalize common textual variants into L_/R_ names
    low = t.lower().strip()
    # Hand variants
    if low in {"left hand", "l hand", "lhand", "left_hand", "lefthand"}:
        return "L_Hand"
    if low in {"right hand", "r hand", "rhand", "right_hand", "righthand"}:
        return "R_Hand"

    # Map finger bones (L_/R_ finger names) to hands
    if t.startswith("L_") and any(k in t for k in _FINGER_KEYWORDS):
        return "L_Hand"
    if t.startswith("R_") and any(k in t for k in _FINGER_KEYWORDS):
        return "R_Hand"

    # Map "Left X" / "Right X" to L_/R_ format
    # Also handle inputs like left_wrist, Left-Wrist, right wrist, etc.
    side = None
    base = None
    tmp = low.replace("_", " ").replace("-", " ")
    if tmp.startswith("left "):
        side = "L"
        base = tmp[5:].strip()
    elif tmp.startswith("right "):
        side = "R"
        base = tmp[6:].strip()

    if side and base:
        base_title = base.title().replace(
            " ", ""
        )  # Wrist -> Wrist, toe base -> Toebase (not used here)
        candidate = f"{side}_{base_title}"
        if candidate in ALLOWED_CONTACT_JOINTS:
            return candidate

    return None


def _build_prompt(object_name: str, allowed_joints: Sequence[str]) -> str:
    allowed_str = ", ".join(allowed_joints)
    return (
        "You are a careful vision classifier. "
        "Given an image of a person and an object, "
        "select which human body joint(s) are in contact with the object. "
        "Use only names from the allowed joint list.\n\n"
        f"Object: {object_name}\n"
        f"Allowed joints: [{allowed_str}]\n\n"
        "Guidelines:\n"
        "- Choose one or more joints that are in contact with the object.\n"
        "- Contact means the person's body part is physically touching or holding the object.\n"
        "- Near proximity without touching does NOT count as contact.\n"
        "- If no body part is in contact with the object, return an empty list.\n"
        "- Order the list so the primary/most direct contact joint appears first.\n"
        "- If fingers are used, select L_Hand or R_Hand (do not list finger bones).\n"
        "- Prefer the most specific joints (e.g., L_Wrist or L_Hand rather than L_Arm).\n"
        "- If multiple joints are clearly in contact, list them all (limit to 1-4).\n\n"
        "Respond with strictly valid JSON using this schema (no extra text):\n"
        '{"joints": ["<JointName1>", "<JointName2>"]}'
    )


def _parse_joints_from_response(text: str, allowed_joints: Sequence[str]) -> list[str]:
    """Parse assistant response and return a filtered, unique list of joints (order preserved)."""

    def _normalize_collect(seq: Sequence[str]) -> list[str]:
        mapped: list[str] = []
        for s in seq:
            if isinstance(s, str):
                m = _token_to_allowed(s)
                if m is not None and m in allowed_joints:
                    mapped.append(m)
        return _unique_preserve_order(mapped)

    # Try strict JSON load first
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict) and isinstance(loaded.get("joints"), list):
            return _normalize_collect(loaded["joints"])
        if isinstance(loaded, list):
            return _normalize_collect(loaded)
    except Exception:
        pass

    # Fallback: extract the first JSON object or array via regex
    obj_match = re.search(r"\{[\s\S]*\}", text)
    arr_match = re.search(r"\[[\s\S]*\]", text)
    snippet = obj_match.group(0) if obj_match else (arr_match.group(0) if arr_match else None)
    if snippet:
        try:
            loaded = json.loads(snippet)
            if isinstance(loaded, dict) and isinstance(loaded.get("joints"), list):
                return _normalize_collect(loaded["joints"])
            if isinstance(loaded, list):
                return _normalize_collect(loaded)
        except Exception:
            pass

    # Final fallback: try to find quoted tokens that look like joints
    rough = re.findall(r'"([A-Za-z_ ]+)"', text)
    return _normalize_collect(rough)


def detect_interaction_joints(
    image_path: str,
    object_name: str,
    *,
    allowed_joints: Sequence[str] | None = None,
    model: str = DEFAULT_REASONING_MODEL,
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> list[str]:
    """
    Identify which body joint(s) interact with the given object in the image.

    Args:
        image_path: Path to the interaction image.
        object_name: Human-readable object name/category (e.g., "chair").
        allowed_joints: Optional override list of joint names to choose from.
        model: OpenAI model to use.
        max_tokens: Max tokens for the completion.
        temperature: Sampling temperature (use low value for determinism).

    Returns:
        Ordered list of selected joint names (primary first), subset of allowed joints.
    """
    allowed = list(allowed_joints) if allowed_joints is not None else list(ALLOWED_CONTACT_JOINTS)

    system_prompt = (
        "You are a precise multi-modal classifier. "
        "Follow the schema exactly and only use allowed joint names."
    )
    user_prompt = _build_prompt(object_name=object_name, allowed_joints=allowed)

    raw = chat_with_image(
        prompt_text=user_prompt,
        image_path=image_path,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        response_format={"type": "json_object"},
    )

    return _parse_joints_from_response(raw, allowed)


def detect_interaction(
    image_path: str,
    object_name: str,
    *,
    model: str = DEFAULT_REASONING_MODEL,
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> bool:
    """
    Determine whether the human is in contact with the given object in the image.

    Args:
        image_path: Path to the interaction image.
        object_name: Human-readable object name/category (e.g., "chair", "bottle").
        model: OpenAI model to use.
        max_tokens: Max tokens for the completion.
        temperature: Sampling temperature (use low value for determinism).

    Returns:
        True if the human is in contact with the object, False otherwise.
    """
    system_prompt = (
        "You are a precise multi-modal vision classifier. "
        "Follow the schema exactly and provide accurate binary judgments."
    )

    user_prompt = (
        "You are a careful vision classifier. "
        "Given an image of a person and an object, "
        "determine whether the person is in physical contact with the object.\n\n"
        f"Object: {object_name}\n\n"
        "Guidelines:\n"
        "- Contact means the person is physically touching or holding the object.\n"
        "- Near proximity without touching does NOT count as contact.\n"
        "- If any part of the person's body touches the object, return true.\n"
        "- Be conservative: when uncertain, return false.\n\n"
        "Respond with strictly valid JSON using this schema (no extra text):\n"
        '{"in_contact": true/false, "confidence": "high/medium/low"}'
    )

    raw = chat_with_image(
        prompt_text=user_prompt,
        image_path=image_path,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        response_format={"type": "json_object"},
    )

    # Parse the response
    in_contact = _parse_contact_from_response(raw)
    return in_contact


def _parse_contact_from_response(text: str) -> bool:
    """Parse assistant response and return boolean indicating contact."""
    # Try strict JSON load first
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            contact_value = loaded.get("in_contact")
            if isinstance(contact_value, bool):
                return contact_value
            # Handle string representations
            if isinstance(contact_value, str):
                return contact_value.lower() in {"true", "yes", "1"}
    except Exception:
        pass

    # Fallback: extract the first JSON object via regex
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            loaded = json.loads(obj_match.group(0))
            if isinstance(loaded, dict):
                contact_value = loaded.get("in_contact")
                if isinstance(contact_value, bool):
                    return contact_value
                if isinstance(contact_value, str):
                    return contact_value.lower() in {"true", "yes", "1"}
        except Exception:
            pass

    # Final fallback: search for boolean keywords in the text
    text_lower = text.lower()
    if "true" in text_lower or '"in_contact": true' in text_lower:
        return True
    if "false" in text_lower or '"in_contact": false' in text_lower:
        return False

    # Default to False if we can't parse
    return False


def detect_contact_joints_interval(
    image_list: list[str],
    object_name: str,
    interval_length: int,
    start_idx: int = 0,
    end_idx: int | None = None,
    *,
    allowed_joints: Sequence[str] | None = None,
    model: str = DEFAULT_REASONING_MODEL,
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> list[list[str] | None]:
    """
    Detect contact labels per interval instead of using a single label for the
    whole sequence.  For each interval a representative frame (the middle frame)
    is sent to the VLM.  If the model determines the person is **in contact**
    with the object, `detect_interaction_joints` is called on that frame to
    obtain the joint list; otherwise the interval's entry is ``None``.

    Args:
        image_list: Ordered list of frame image paths.
        object_name: Human-readable object name/category.
        interval_length: Number of frames per interval.
        start_idx: First frame index to consider (inclusive).
        end_idx: Last frame index to consider (exclusive). Defaults to
            ``len(image_list)``.
        allowed_joints: Optional override list of joint names.
        model: OpenAI model to use.
        max_tokens: Max tokens for each completion.
        temperature: Sampling temperature.

    Returns:
        A list with one entry per interval.  Each entry is either a list of
        joint-name strings (primary first) when the person is in contact, or
        ``None`` when no contact is detected.
    """
    if end_idx is None:
        end_idx = len(image_list)

    contact_labels_per_interval: list[list[str] | None] = []

    idx = start_idx
    while idx < end_idx:
        interval_end = min(idx + interval_length, end_idx)
        # Use the middle frame of the interval as representative
        mid = (idx + interval_end) // 2
        image_path = image_list[mid]

        # Step 1: check if there is any contact in this interval
        in_contact = detect_interaction(
            image_path,
            object_name,
            model=model,
            max_tokens=16384,
            temperature=temperature,
        )

        if in_contact:
            # Step 2: identify which joints are in contact
            joints = detect_interaction_joints(
                image_path,
                object_name,
                allowed_joints=allowed_joints,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            contact_labels_per_interval.append(joints if joints else None)
        else:
            contact_labels_per_interval.append(None)

        idx = interval_end

    return contact_labels_per_interval


def _as_mask_array(masks, frame_num: int) -> np.ndarray:
    import numpy as np

    arr = np.asarray(masks).astype(bool)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3 or arr.shape[0] != frame_num:
        raise ValueError(f"masks must have shape [T,H,W], got {arr.shape}")
    return arr


def detect_contact_labels_from_masks(
    hand_keypoints_2d,
    human_masks,
    object_masks,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
    interval_length: int = 8,
    hand: str = "auto",
    distance_threshold_px: float = 12.0,
    min_mask_iou: float = 0.001,
    min_hand_inside_ratio: float = 0.20,
    required_consecutive: int = 2,
) -> tuple[list[list[str] | None], int, dict]:
    """Compute contact labels without a VLM.

    A frame is contact-positive when the selected hand has keypoints close to
    or inside the object mask and the human/object masks overlap (or the hand
    has sufficient in-mask coverage).  The earliest sustained positive frame
    is returned and labels are serialized in the legacy per-interval format.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, distance_transform_edt

    points = np.asarray(hand_keypoints_2d, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] < 2 or points.shape[1] % 2:
        raise ValueError(f"hand_keypoints_2d must be [T,32,2/3], got {points.shape}")
    if points.shape[-1] == 2:
        points = np.concatenate([points, np.ones((*points.shape[:2], 1))], axis=-1)
    frame_num = points.shape[0]
    human = _as_mask_array(human_masks, frame_num)
    obj = _as_mask_array(object_masks, frame_num)
    end_idx = frame_num if end_idx is None else min(frame_num, int(end_idx))
    start_idx = max(0, int(start_idx))
    if end_idx <= start_idx:
        return [], start_idx, {"contact_frames": [], "mask_iou": [], "left_distance_px": [], "right_distance_px": []}
    hand = str(hand).lower()
    if hand not in {"auto", "left", "right", "both"}:
        raise ValueError("contact.hand must be auto, left, right, or both")
    split = points.shape[1] // 2
    side_points = {"left": points[:, :split], "right": points[:, split:]}
    distances = {"left": np.full(frame_num, np.inf), "right": np.full(frame_num, np.inf)}
    inside = {"left": np.zeros(frame_num), "right": np.zeros(frame_num)}
    mask_iou = np.zeros(frame_num, dtype=np.float64)
    for i in range(frame_num):
        union = np.logical_or(human[i], obj[i]).sum()
        mask_iou[i] = np.logical_and(human[i], obj[i]).sum() / max(float(union), 1.0)
        dilated = binary_dilation(obj[i], iterations=2)
        distance = distance_transform_edt(~dilated)
        for side, side_kp in side_points.items():
            valid = np.isfinite(side_kp[i, :, :2]).all(axis=1) & (side_kp[i, :, 2] >= 0.2)
            if not valid.any():
                continue
            xy = np.rint(side_kp[i, valid, :2]).astype(np.int64)
            in_bounds = (
                (xy[:, 0] >= 0) & (xy[:, 0] < obj.shape[2]) &
                (xy[:, 1] >= 0) & (xy[:, 1] < obj.shape[1])
            )
            xy = xy[in_bounds]
            if xy.size == 0:
                continue
            distances[side][i] = float(np.median(distance[xy[:, 1], xy[:, 0]]))
            inside[side][i] = float(np.mean(dilated[xy[:, 1], xy[:, 0]]))

    def positive(side):
        return (distances[side] <= float(distance_threshold_px)) & (
            (mask_iou >= float(min_mask_iou)) |
            (inside[side] >= float(min_hand_inside_ratio))
        )

    positives = {side: positive(side) for side in ("left", "right")}
    if hand == "both":
        frame_positive = positives["left"] & positives["right"]
        label_sides = ("left", "right")
    elif hand in ("left", "right"):
        frame_positive = positives[hand]
        label_sides = (hand,)
    else:
        frame_positive = positives["left"] | positives["right"]
        label_sides = ("left", "right")

    run = max(1, int(required_consecutive))
    contact_start = end_idx
    for i in range(start_idx, max(start_idx, end_idx - run + 1)):
        if bool(np.all(frame_positive[i : i + run])):
            contact_start = i
            break
    if contact_start >= end_idx:
        # Preserve a usable cache even when masks are noisy; object motion onset
        # remains the physically computed fallback supplied by the caller.
        contact_start = start_idx
        frame_positive[start_idx:end_idx] = True
        if hand == "both":
            positives["left"][start_idx:end_idx] = True
            positives["right"][start_idx:end_idx] = True
        elif hand in ("left", "right"):
            positives[hand][start_idx:end_idx] = True
        else:
            positives["right"][start_idx:end_idx] = True

    labels: list[list[str] | None] = []
    idx = start_idx
    while idx < end_idx:
        interval_end = min(end_idx, idx + max(1, int(interval_length)))
        if interval_end <= contact_start:
            labels.append(None)
        else:
            active = [s for s in label_sides if np.any(positives[s][max(idx, contact_start):interval_end])]
            if hand == "both":
                active = ["left", "right"] if np.any(frame_positive[max(idx, contact_start):interval_end]) else []
            labels.append(["L_Hand" if s == "left" else "R_Hand" for s in active] or None)
        idx = interval_end
    diagnostics = {
        "contact_frames": np.flatnonzero(frame_positive).astype(int).tolist(),
        "mask_iou": mask_iou.astype(float).tolist(),
        "left_distance_px": distances["left"].astype(float).tolist(),
        "right_distance_px": distances["right"].astype(float).tolist(),
    }
    return labels, int(contact_start), diagnostics
