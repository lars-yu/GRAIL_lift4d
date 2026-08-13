#!/usr/bin/env python3
"""Enrich a static-only VGGT scene reference with legacy first-frame anchors.

This is useful when the Blender static scene export intentionally excludes the
manipulated object/human, but an existing FoundationPose preparation directory
still has the first-frame pose and masks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from grail.core.io import load_init_rendering_data


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _write_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(metadata, handle, indent=2)


def _legacy_object_pose(first_frame_pose: Path) -> tuple[np.ndarray, list[float]]:
    obj_R, obj_t, obj_scale, _, _, _ = load_init_rendering_data(str(first_frame_pose), device="cpu")
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(obj_R, dtype=np.float32).reshape(3, 3)
    pose[:3, 3] = np.asarray(obj_t, dtype=np.float32).reshape(3)
    scale = np.asarray(obj_scale, dtype=np.float32).reshape(-1).tolist()
    return pose, [float(v) for v in scale]


def _copy_file(src: Path | None, dst: Path, overwrite: bool) -> dict:
    result = {
        "source": str(src) if src is not None else None,
        "target": str(dst),
        "copied": False,
        "exists": dst.exists(),
    }
    if src is None:
        result["reason"] = "source not provided"
        return result
    src = Path(src)
    if not src.exists():
        result["reason"] = "source missing"
        return result
    if dst.exists() and not overwrite:
        result["reason"] = "target exists"
        result["exists"] = True
        return result
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    result["copied"] = True
    result["exists"] = True
    return result


def _write_static_mask_from_foreground(
    human_mask_path: Path,
    object_mask_path: Path,
    static_mask_path: Path,
    overwrite: bool,
) -> dict:
    result = {
        "human_mask": str(human_mask_path),
        "object_mask": str(object_mask_path),
        "target": str(static_mask_path),
        "written": False,
        "exists": static_mask_path.exists(),
    }
    if static_mask_path.exists() and not overwrite:
        result["reason"] = "target exists"
        return result
    human_mask = cv2.imread(str(human_mask_path), cv2.IMREAD_GRAYSCALE)
    object_mask = cv2.imread(str(object_mask_path), cv2.IMREAD_GRAYSCALE)
    if human_mask is None or object_mask is None:
        result["reason"] = "foreground mask missing or unreadable"
        return result
    if human_mask.shape != object_mask.shape:
        result["reason"] = (
            f"foreground mask shape mismatch: human={human_mask.shape}, object={object_mask.shape}"
        )
        return result

    static_mask = (~((human_mask > 0) | (object_mask > 0))).astype(np.uint8) * 255
    static_mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(static_mask_path), static_mask):
        result["reason"] = "cv2.imwrite failed"
        return result
    result.update(
        {
            "written": True,
            "exists": True,
            "shape": list(static_mask.shape),
            "static_pixels": int((static_mask > 0).sum()),
            "foreground_pixels": int((static_mask == 0).sum()),
            "formula": "NOT (human_mask OR object_mask)",
        }
    )
    return result


def _safe_relative_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        return Path(path.name)
    return path


def _obj_materials(obj_path: Path) -> list[Path]:
    materials = []
    try:
        with open(obj_path, "r", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2 and parts[0] == "mtllib":
                    materials.append(_safe_relative_path(parts[1]))
    except OSError:
        pass
    return materials


def _mtl_textures(mtl_path: Path) -> list[Path]:
    textures = []
    texture_keys = {
        "map_Ka",
        "map_Kd",
        "map_Ks",
        "map_Ke",
        "map_Ns",
        "map_d",
        "map_Bump",
        "map_bump",
        "bump",
        "disp",
        "decal",
    }
    try:
        with open(mtl_path, "r", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] in texture_keys:
                    textures.append(_safe_relative_path(parts[-1]))
    except OSError:
        pass
    return textures


def _copy_obj_sidecars(obj_src: Path, scene_reference_dir: Path, overwrite: bool) -> list[dict]:
    copied = []
    for material_rel in _obj_materials(obj_src):
        material_src = obj_src.parent / material_rel
        if not material_src.exists():
            same_stem_material = obj_src.with_suffix(".mtl")
            if same_stem_material.exists():
                material_src = same_stem_material
        material_dst = scene_reference_dir / material_rel
        copied.append(_copy_file(material_src, material_dst, overwrite))
        for texture_rel in _mtl_textures(material_src):
            texture_src = material_src.parent / texture_rel
            texture_dst = material_dst.parent / texture_rel
            copied.append(_copy_file(texture_src, texture_dst, overwrite))
    return copied


def enrich_scene_reference(
    scene_reference_dir,
    first_frame_pose,
    *,
    human_mask=None,
    object_mask=None,
    human_mesh=None,
    object_mesh=None,
    overwrite=False,
    copy_mesh_sidecars=True,
) -> dict:
    scene_reference_dir = Path(scene_reference_dir)
    first_frame_pose = Path(first_frame_pose)
    scene_reference_dir.mkdir(parents=True, exist_ok=True)

    pose, object_scale = _legacy_object_pose(first_frame_pose)
    object_init_pose = scene_reference_dir / "object_init_pose.npy"
    wrote_object_pose = False
    if overwrite or not object_init_pose.exists():
        np.save(object_init_pose, pose)
        wrote_object_pose = True

    copies = {
        "human_mask": _copy_file(Path(human_mask) if human_mask else None, scene_reference_dir / "human_mask_00000.png", overwrite),
        "object_mask": _copy_file(Path(object_mask) if object_mask else None, scene_reference_dir / "object_mask_00000.png", overwrite),
        "human_obj": _copy_file(Path(human_mesh) if human_mesh else None, scene_reference_dir / "human_init.obj", overwrite),
        "object_obj": _copy_file(Path(object_mesh) if object_mesh else None, scene_reference_dir / "object.obj", overwrite),
    }
    static_mask = _write_static_mask_from_foreground(
        scene_reference_dir / "human_mask_00000.png",
        scene_reference_dir / "object_mask_00000.png",
        scene_reference_dir / "static_mask_00000.png",
        overwrite,
    )
    sidecars = []
    if human_mesh and copy_mesh_sidecars:
        sidecars.extend(_copy_obj_sidecars(Path(human_mesh), scene_reference_dir, overwrite))
    if object_mesh and copy_mesh_sidecars:
        sidecars.extend(_copy_obj_sidecars(Path(object_mesh), scene_reference_dir, overwrite))

    metadata_path = scene_reference_dir / "metadata.json"
    metadata = _load_metadata(metadata_path)
    exports = metadata.setdefault("exports", {})
    exports["object_init_pose"] = object_init_pose.exists()
    for key, filename in (
        ("static_mask", "static_mask_00000.png"),
        ("human_mask", "human_mask_00000.png"),
        ("object_mask", "object_mask_00000.png"),
        ("human_init_obj", "human_init.obj"),
        ("object_obj", "object.obj"),
    ):
        if (scene_reference_dir / filename).exists():
            exports[key] = True

    metadata["legacy_enrichment"] = {
        "first_frame_pose": str(first_frame_pose),
        "object_init_pose": str(object_init_pose),
        "object_pose_source": "first_frame_pose.pickle obj_R/obj_t",
        "object_scale": object_scale,
        "object_scale_note": "Scale is recorded for audit only and is not folded into object_init_pose.npy.",
        "human_mask_source": str(human_mask) if human_mask else None,
        "object_mask_source": str(object_mask) if object_mask else None,
        "static_mask": static_mask,
        "human_mesh_source": str(human_mesh) if human_mesh else None,
        "object_mesh_source": str(object_mesh) if object_mesh else None,
        "coordinate_space": "blender_world_T_B<-O",
    }
    _write_metadata(metadata_path, metadata)

    return {
        "scene_reference_dir": str(scene_reference_dir),
        "object_init_pose": str(object_init_pose),
        "wrote_object_init_pose": wrote_object_pose,
        "object_scale": object_scale,
        "copies": copies,
        "static_mask": static_mask,
        "sidecars": sidecars,
        "exports": exports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich a static-only VGGT scene_reference with legacy first-frame anchors"
    )
    parser.add_argument("--scene_reference_dir", required=True)
    parser.add_argument("--first_frame_pose", required=True)
    parser.add_argument("--human_mask", default=None)
    parser.add_argument("--object_mask", default=None)
    parser.add_argument("--human_mesh", default=None)
    parser.add_argument("--object_mesh", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_mesh_sidecars", action="store_true")
    args = parser.parse_args()

    summary = enrich_scene_reference(
        args.scene_reference_dir,
        args.first_frame_pose,
        human_mask=args.human_mask,
        object_mask=args.object_mask,
        human_mesh=args.human_mesh,
        object_mesh=args.object_mesh,
        overwrite=args.overwrite,
        copy_mesh_sidecars=not args.no_mesh_sidecars,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
