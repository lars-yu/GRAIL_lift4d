#!/usr/bin/env python3
"""Export Blender scene anchors for VGGT dynamic-camera alignment.

Usage:
    blender -b scene.blend --python scripts/export_scene_reference.py -- \
        --output_dir scene_reference \
        --object_regex "object|target" \
        --human_regex "human|character|person"

The static scene export deliberately excludes objects matching the human/object
regexes so dynamic foreground geometry is not used as the main Sim(3) anchor.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def _argv():
    argv = sys.argv
    return argv[argv.index("--") + 1 :] if "--" in argv else []


def _match(obj, pattern: str | None) -> bool:
    return bool(pattern and re.search(pattern, obj.name, flags=re.IGNORECASE))


def _mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def _select(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select_set(True)
    if objs:
        bpy.context.view_layer.objects.active = objs[0]


def _export_obj(path: Path, objs) -> bool:
    if not objs:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    _select(objs)
    if hasattr(bpy.ops.wm, "obj_export"):
        # Keep exported OBJ vertices in the same Blender XYZ metric world
        # convention as static_scene.ply. Blender 5's exporter otherwise
        # defaults to a Y-up/negative-Z-forward interchange convention.
        bpy.ops.wm.obj_export(
            filepath=str(path),
            export_selected_objects=True,
            forward_axis="Y",
            up_axis="Z",
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=str(path),
            use_selection=True,
            axis_forward="Y",
            axis_up="Z",
        )
    return True


def _export_ply(path: Path, objs, sample_spacing_m: float = 0.02) -> bool:
    if not objs:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    chunks = []
    vertex_count = 0
    sample_spacing_m = float(sample_spacing_m)
    if sample_spacing_m < 0:
        raise ValueError(f"sample_spacing_m must be non-negative, got {sample_spacing_m}")
    for object_idx, obj in enumerate(objs):
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            if mesh is None or len(mesh.vertices) == 0:
                continue
            coords = np.empty((len(mesh.vertices), 3), dtype=np.float32)
            mesh.vertices.foreach_get("co", coords.reshape(-1))
            matrix = np.asarray(eval_obj.matrix_world, dtype=np.float32)
            coords = coords @ matrix[:3, :3].T + matrix[:3, 3]
            chunks.append(coords.astype("<f4", copy=False))
            vertex_count += coords.shape[0]

            # Mesh vertices alone leave large floors and walls represented only
            # by their corners. Add deterministic surface samples so the PLY is
            # a usable point-to-surface reference for alignment validation.
            if sample_spacing_m > 0 and len(mesh.polygons) > 0:
                mesh.calc_loop_triangles()
                triangle_count = len(mesh.loop_triangles)
                if triangle_count > 0:
                    triangle_indices = np.empty((triangle_count, 3), dtype=np.int32)
                    mesh.loop_triangles.foreach_get("vertices", triangle_indices.reshape(-1))
                    triangles = coords[triangle_indices]
                    triangle_areas = 0.5 * np.linalg.norm(
                        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                        axis=1,
                    )
                    sample_counts = np.floor(
                        triangle_areas / (sample_spacing_m * sample_spacing_m)
                    ).astype(np.int64)
                    rng = np.random.default_rng(object_idx)
                    sampled_chunks = []
                    for triangle, count in zip(triangles, sample_counts):
                        if count <= 0:
                            continue
                        count = min(int(count), 500_000)
                        uv = rng.random((count, 2), dtype=np.float32)
                        folded = uv.sum(axis=1) > 1.0
                        uv[folded] = 1.0 - uv[folded]
                        samples = (
                            triangle[0]
                            + uv[:, :1] * (triangle[1] - triangle[0])
                            + uv[:, 1:] * (triangle[2] - triangle[0])
                        )
                        sampled_chunks.append(samples.astype("<f4", copy=False))
                    if sampled_chunks:
                        samples = np.concatenate(sampled_chunks, axis=0)
                        chunks.append(samples)
                        vertex_count += samples.shape[0]
        finally:
            eval_obj.to_mesh_clear()

    if vertex_count == 0:
        return False

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as handle:
        handle.write(header)
        for coords in chunks:
            handle.write(coords.tobytes())
    return True


def _save_transform(path: Path, matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(matrix, dtype=np.float32))


def _camera_c2w(camera_obj) -> np.ndarray:
    return np.asarray(camera_obj.matrix_world, dtype=np.float32)


def _camera_K(camera_obj, width: int, height: int) -> np.ndarray:
    data = camera_obj.data
    sensor_width = data.sensor_width
    sensor_height = data.sensor_height
    if data.sensor_fit == "VERTICAL":
        sensor_width = sensor_height * (width / height)
    else:
        sensor_height = sensor_width * (height / width)
    fx = (data.lens / sensor_width) * width
    fy = (data.lens / sensor_height) * height
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _apply_camera_pose_pickle(camera_obj, path: str | Path) -> dict:
    path = Path(path)
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Camera pose pickle must contain a dict, got {type(data).__name__}")

    rotation = data.get("R", data.get("cam_R"))
    translation = data.get("t", data.get("cam_t"))
    if rotation is None or translation is None:
        raise KeyError("Camera pose pickle must contain R/t or cam_R/cam_t")
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(
            f"Invalid camera pose shapes: rotation={rotation.shape}, translation={translation.shape}"
        )

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = translation
    camera_obj.matrix_world = Matrix(c2w.tolist())

    scene = bpy.context.scene
    resolution = data.get("resolution")
    if resolution is None and data.get("frame_width") is not None and data.get("frame_height") is not None:
        resolution = (data["frame_width"], data["frame_height"])
    if resolution is not None:
        width, height = (int(resolution[0]), int(resolution[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid camera resolution: {(width, height)}")
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = 1.0
        scene.render.pixel_aspect_y = 1.0
    else:
        width = int(scene.render.resolution_x)
        height = int(scene.render.resolution_y)

    focal_px = data.get("focal_length")
    if focal_px is not None:
        focal_px = float(focal_px)
        if not np.isfinite(focal_px) or focal_px <= 0:
            raise ValueError(f"Invalid focal_length in camera pose pickle: {focal_px}")
        if width >= height:
            camera_obj.data.sensor_fit = "HORIZONTAL"
            camera_obj.data.lens = focal_px * float(camera_obj.data.sensor_width) / float(width)
        else:
            camera_obj.data.sensor_fit = "VERTICAL"
            camera_obj.data.lens = focal_px * float(camera_obj.data.sensor_height) / float(height)

    bpy.context.view_layer.update()
    return {
        "source": str(path.resolve()),
        "rotation_key": "R" if "R" in data else "cam_R",
        "translation_key": "t" if "t" in data else "cam_t",
        "resolution": [width, height],
        "focal_length_px": float(focal_px) if focal_px is not None else None,
        "coordinate_convention": "Blender camera matrix_world (camera-to-Blender-world)",
    }


def _raycast_camera_z_depth(path: Path) -> None:
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise RuntimeError("Cannot render depth without an active camera")
    width = int(scene.render.resolution_x)
    height = int(scene.render.resolution_y)
    K = _camera_K(camera, width, height)
    cam_world = camera.matrix_world
    cam_world_inv = cam_world.inverted()
    origin = cam_world.translation
    rotation = cam_world.to_3x3()
    depth = np.zeros((height, width), dtype=np.float32)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    for y in range(height):
        y_cv = (y + 0.5 - K[1, 2]) / K[1, 1]
        for x in range(width):
            x_cv = (x + 0.5 - K[0, 2]) / K[0, 0]
            # Blender camera local axes: +X right, +Y up, -Z forward.
            direction_local = Vector((x_cv, -y_cv, -1.0)).normalized()
            direction_world = (rotation @ direction_local).normalized()
            hit, location, _, _, _, _ = scene.ray_cast(depsgraph, origin, direction_world)
            if hit:
                location_camera = cam_world_inv @ location
                depth[y, x] = max(float(-location_camera.z), 0.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    image = bpy.data.images.new(path.stem, width, height, alpha=True, float_buffer=True)
    try:
        image.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    pixels = np.zeros((height, width, 4), dtype=np.float32)
    pixels[..., :3] = depth[..., None]
    pixels[..., 3] = 1.0
    # Blender image buffers are bottom-up; OpenCV reads the resulting EXR with
    # a top-left origin. Non-Color preserves the metric values during saving.
    image.pixels.foreach_set(np.flipud(pixels).reshape(-1))
    image.filepath_raw = str(path)
    try:
        image.file_format = "OPEN_EXR"
    except TypeError:
        image.file_format = "OPEN_EXR_MULTILAYER"
    image.save()
    bpy.data.images.remove(image)


def _normalize_depth_pass_to_opencv_camera_z(path: Path) -> None:
    """Store Blender compositor camera-Z with OpenCV's top-left origin."""
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise RuntimeError("Cannot convert depth without an active camera")
    source = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = (int(source.size[0]), int(source.size[1]))
        channels = int(source.channels)
        raw = np.empty(width * height * channels, dtype=np.float32)
        source.pixels.foreach_get(raw)
        # Blender image pixels are bottom-up while OpenCV and all GRAIL masks
        # use a top-left origin. The compositor Depth/Z pass is already
        # camera-Z, as verified against scene.ray_cast camera coordinates.
        camera_z = np.flipud(raw.reshape(height, width, channels)[..., 0]).copy()
    finally:
        bpy.data.images.remove(source)

    valid = (
        np.isfinite(camera_z)
        & (camera_z > 0)
        & (camera_z <= float(camera.data.clip_end) * 1.01)
    )
    camera_z[~valid] = 0.0

    image = bpy.data.images.new(path.stem, width, height, alpha=True, float_buffer=True)
    try:
        image.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    pixels = np.zeros((height, width, 4), dtype=np.float32)
    pixels[..., :3] = camera_z[..., None]
    pixels[..., 3] = 1.0
    image.pixels.foreach_set(np.flipud(pixels).reshape(-1))
    image.filepath_raw = str(path)
    try:
        image.file_format = "OPEN_EXR"
    except TypeError:
        image.file_format = "OPEN_EXR_MULTILAYER"
    image.save()
    bpy.data.images.remove(image)


def _render_depth(path: Path) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.view_layers[0].use_pass_z = True
    path = path.with_suffix(".exr")
    path.parent.mkdir(parents=True, exist_ok=True)

    old_use_nodes = scene.use_nodes
    old_compositor_group = getattr(scene, "compositing_node_group", None)
    old_filepath = scene.render.filepath
    old_file_format = scene.render.image_settings.file_format
    scene.use_nodes = True
    owned_tree = None
    if hasattr(scene, "node_tree"):
        tree = scene.node_tree
    else:
        tree = getattr(scene, "compositing_node_group", None)
        if tree is None:
            tree = bpy.data.node_groups.new("GRAIL_depth_compositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
            owned_tree = tree
    render_layers = None
    output = None
    try:
        render_layers = tree.nodes.new(type="CompositorNodeRLayers")
        output = tree.nodes.new(type="CompositorNodeOutputFile")
        if hasattr(output, "base_path"):
            output.base_path = str(path.parent)
            output.file_slots[0].path = path.stem
        else:
            output.directory = str(path.parent)
            output.file_name = path.stem
        try:
            output.format.file_format = "OPEN_EXR"
        except TypeError:
            output.format.file_format = "OPEN_EXR_MULTILAYER"
        depth_socket = render_layers.outputs.get("Depth") or render_layers.outputs.get("Z")
        if depth_socket is None:
            raise RuntimeError("Blender render layer has no Depth/Z output socket")
        tree.links.new(depth_socket, output.inputs[0])
        bpy.ops.render.render(write_still=False)

        produced = sorted(path.parent.glob(f"{path.stem}*.exr"))
        if produced:
            produced[-1].replace(path)
    finally:
        for node in (output, render_layers):
            if node is not None:
                try:
                    tree.nodes.remove(node)
                except Exception:
                    pass
        scene.render.filepath = old_filepath
        scene.render.image_settings.file_format = old_file_format
        scene.use_nodes = old_use_nodes
        if hasattr(scene, "compositing_node_group"):
            scene.compositing_node_group = old_compositor_group
        if owned_tree is not None:
            try:
                bpy.data.node_groups.remove(owned_tree)
            except Exception:
                pass
    if path.exists():
        _normalize_depth_pass_to_opencv_camera_z(path)
    else:
        _raycast_camera_z_depth(path)


def _render_binary_mask(path: Path, objs) -> bool:
    if not objs:
        return False
    original_materials = {obj.name: list(obj.data.materials) for obj in _mesh_objects()}
    white = bpy.data.materials.new("GRAIL_mask_white")
    white.diffuse_color = (1, 1, 1, 1)
    black = bpy.data.materials.new("GRAIL_mask_black")
    black.diffuse_color = (0, 0, 0, 1)
    selected = set(obj.name for obj in objs)
    for obj in _mesh_objects():
        obj.data.materials.clear()
        obj.data.materials.append(white if obj.name in selected else black)
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    for obj in _mesh_objects():
        obj.data.materials.clear()
        for mat in original_materials[obj.name]:
            obj.data.materials.append(mat)
    return True


def _sample_names(objs, limit: int = 20) -> list[str]:
    names = [obj.name for obj in objs]
    return names[:limit]


def _strict_check(args, exports: dict, meshes) -> None:
    if not args.strict:
        return
    missing = []
    required = [
        ("static_scene.obj", exports.get("static_scene_obj")),
        ("static_scene.ply", exports.get("static_scene_ply")),
        ("camera_init_c2w.npy", exports.get("camera_init_c2w")),
        ("camera_init_K.npy", exports.get("camera_init_K")),
    ]
    if args.render_depth:
        required.append(("depth_gt_00000.exr", exports.get("depth_gt_00000")))
    if not args.allow_missing_human:
        required.append(("human_init.obj", exports.get("human_init_obj")))
        if args.render_masks:
            required.append(("human_mask_00000.png", exports.get("human_mask")))
    if not args.allow_missing_object:
        required.append(("object.obj", exports.get("object_obj")))
        required.append(("object_init_pose.npy", exports.get("object_init_pose")))
        if args.render_masks:
            required.append(("object_mask_00000.png", exports.get("object_mask")))
    if args.render_masks:
        required.append(("static_mask_00000.png", exports.get("static_mask")))
    for name, ok in required:
        if not ok:
            missing.append(name)
    if missing:
        print(
            "Strict scene reference export missing required artifacts: "
            + ", ".join(missing)
            + f". human_regex={args.human_regex!r}, object_regex={args.object_regex!r}. "
            + "Use --human_regex/--object_regex or --allow_missing_human/--allow_missing_object. "
            + f"Sample mesh names: {_sample_names(meshes)}",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _strict_preflight(args, meshes, static, human, obj, camera) -> None:
    if not args.strict:
        return
    missing = []
    if not static:
        missing.append("static mesh group")
    if camera is None:
        missing.append("camera")
    if not args.allow_missing_human and not human:
        missing.append("human mesh group")
    if not args.allow_missing_object and not obj:
        missing.append("object mesh group")
    if missing:
        print(
            "Strict scene reference export preflight failed: "
            + ", ".join(missing)
            + f". human_regex={args.human_regex!r}, object_regex={args.object_regex!r}. "
            + "Use --human_regex/--object_regex or --allow_missing_human/--allow_missing_object. "
            + f"Sample mesh names: {_sample_names(meshes)}",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Export scene reference for dynamic-camera VGGT alignment")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--object_regex", default="object|target|manipulated")
    parser.add_argument("--human_regex", default="human|character|person|renderpeople|g1")
    parser.add_argument("--exclude_static_regex", default="movable|dynamic")
    parser.add_argument("--camera", default=None)
    parser.add_argument(
        "--camera_pose_pickle",
        default=None,
        help="Sample camera pickle with R/t or cam_R/cam_t, focal_length, and resolution.",
    )
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--render_masks", action="store_true")
    parser.add_argument(
        "--static_ply_sample_spacing_m",
        type=float,
        default=0.02,
        help="Approximate spacing for static-scene surface samples added to the PLY; 0 disables sampling.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if required scene-reference artifacts are missing.",
    )
    parser.add_argument(
        "--allow_missing_human",
        action="store_true",
        help="With --strict, allow scenes that intentionally have no human mesh.",
    )
    parser.add_argument(
        "--allow_missing_object",
        action="store_true",
        help="With --strict, allow scenes that intentionally have no manipulated object mesh.",
    )
    args = parser.parse_args(_argv())

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meshes = _mesh_objects()
    human = [obj for obj in meshes if _match(obj, args.human_regex)]
    obj = [obj for obj in meshes if _match(obj, args.object_regex)]
    excluded = set(o.name for o in human + obj)
    static = [
        o
        for o in meshes
        if o.name not in excluded and not _match(o, args.exclude_static_regex)
    ]

    camera = bpy.data.objects.get(args.camera) if args.camera else bpy.context.scene.camera
    camera_pose_override = None
    if args.camera_pose_pickle:
        if camera is None:
            raise RuntimeError("--camera_pose_pickle requires an active or named Blender camera")
        camera_pose_override = _apply_camera_pose_pickle(camera, args.camera_pose_pickle)
    _strict_preflight(args, meshes, static, human, obj, camera)

    exports = {}
    exports["static_scene_obj"] = _export_obj(out / "static_scene.obj", static)
    exports["static_scene_ply"] = _export_ply(
        out / "static_scene.ply",
        static,
        sample_spacing_m=args.static_ply_sample_spacing_m,
    )
    exports["human_init_obj"] = _export_obj(out / "human_init.obj", human)
    exports["object_obj"] = _export_obj(out / "object.obj", obj)

    if obj:
        _save_transform(out / "object_init_pose.npy", obj[0].matrix_world)
        exports["object_init_pose"] = True
    else:
        exports["object_init_pose"] = False

    if camera is not None:
        width = int(bpy.context.scene.render.resolution_x)
        height = int(bpy.context.scene.render.resolution_y)
        _save_transform(out / "camera_init_c2w.npy", _camera_c2w(camera))
        np.save(out / "camera_init_K.npy", _camera_K(camera, width, height))
        exports["camera_init_c2w"] = True
        exports["camera_init_K"] = True
        if args.render_depth:
            _render_depth(out / "depth_gt_00000.exr")
            exports["depth_gt_00000"] = (out / "depth_gt_00000.exr").exists()
    else:
        exports["camera_init_c2w"] = False
        exports["camera_init_K"] = False
        exports["depth_gt_00000"] = False

    if args.render_masks:
        exports["static_mask"] = _render_binary_mask(out / "static_mask_00000.png", static)
        exports["human_mask"] = _render_binary_mask(out / "human_mask_00000.png", human)
        exports["object_mask"] = _render_binary_mask(out / "object_mask_00000.png", obj)
    else:
        exports["static_mask"] = False
        exports["human_mask"] = False
        exports["object_mask"] = False

    _strict_check(args, exports, meshes)

    manifest = {
        "static_objects": [o.name for o in static],
        "human_objects": [o.name for o in human],
        "object_objects": [o.name for o in obj],
        "camera": camera.name if camera is not None else None,
        "camera_pose_override": camera_pose_override,
        "object_regex": args.object_regex,
        "human_regex": args.human_regex,
        "exclude_static_regex": args.exclude_static_regex,
        "strict": bool(args.strict),
        "depth_convention": "opencv_camera_z" if exports.get("depth_gt_00000") else None,
        "static_scene_ply_coordinate_space": "blender_metric_world",
        "obj_coordinate_space": "blender_metric_world",
        "obj_export_axes": {"forward": "Y", "up": "Z"},
        "static_ply_sample_spacing_m": float(args.static_ply_sample_spacing_m),
        "exports": exports,
    }
    with open(out / "metadata.json", "w") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
