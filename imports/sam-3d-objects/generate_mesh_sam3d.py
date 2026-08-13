"""
trajectory_folder/
├── frame_0.jpeg
└── masks/
    ├── plane_0.png
    ├── object_0.png
    ├── object_1.png
    ├── object_2.png
    └── ...

python generate_mesh_sam3d.py --image_path /path/to/video_dir/all_frames/000000.png --masks_dir /path/to/video_dir/video_segmentation/masks/frame_000000_masks/

python generate_mesh_sam3d.py --image_path /path/to/video_dir/all_frames/000000.png --masks_dir /path/to/video_dir/video_segmentation/masks/frame_000000_masks/ --with_hand_mesh --mask_files brown_cup_inside_hand_0.png phone_0.png --hand_mask_path /path/to/video_dir/video_segmentation/masks/frame_000000_masks/hand_0.png --hand_mesh_path /path/to/hawor_mesh_0_canonical.obj

python generate_mesh_sam3d.py --image_path /path/to/image.jpg --masks_dir /path/to/masks/

OR, for batch mode:

all_frames_folder/
├── task_frame_0.jpeg
├── task2_frame_0.jpeg
├── masks_task_frame_0/
│   ├── plane_0.png
│   ├── object_0.png
│   ├── object_1.png
│   ├── object_2.png
│   └── ...
└── masks_task2_frame_0/
    ├── plane_0.png
    ├── object_0.png
    ├── object_1.png
    ├── object_2.png
    └── ...


python generate_mesh_sam3d.py --images_dir /path/to/dataset --masks_root /path/to/dataset

OR, for batch mode with a JSON file of image paths:

python generate_mesh_sam3d.py --json_path /path/to/manifest.json --with_mesh_postprocess True --with_texture_baking True --use_vertex_color False
"""


import argparse
import os
import json
from typing import Iterable, List, Optional, Sequence

import torch
import trimesh
from pytorch3d.transforms import quaternion_to_matrix
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
from loguru import logger

from notebook.inference import (  # type: ignore[import-not-found]
    Inference,
    load_image,
    load_mask,
)
from sam3d_objects.data.dataset.tdfy.transforms_3d import (
    compose_transform,
)

import torch.nn.functional as F
import torchvision
from pytorch3d.transforms import Transform3d

from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera
from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap

from omegaconf import OmegaConf
from hydra.utils import instantiate
from pathlib import Path
import open3d as o3d

from tqdm import tqdm


def str2bool(v):
    """Parse boolean values from command line arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


def _iter_images(images_dir: Path, extensions: Sequence[str]) -> Iterable[Path]:
    exts = {e.lower().lstrip(".") for e in extensions if e.strip()}
    for p in sorted(images_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") in exts:
            yield p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM 3D Objects on an RGB image and a set of binary masks "
            "to produce a single 3D scene (Gaussian splats) with all objects "
            "laid out together, like demo_multi_object.ipynb."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=False,
        help="Path to the input RGB image.",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default=None,
        help=(
            "Directory containing the binary masks."
        ),
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help=(
            "If set, run in batch mode over all jpg/jpeg images in this folder. "
            "Each image will use a masks directory of the form: <masks_root>/masks_<img_stem>."
        ),
    )
    parser.add_argument(
        "--masks_root",
        type=str,
        default=None,
        help="Root folder for per-image masks directories in batch mode. Defaults to --images_dir.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to the JSON file containing the image paths and masks directory paths.",
    )
    parser.add_argument(
        "--masks_prefix",
        type=str,
        default="masks_",
        help="Prefix for per-image masks directory in batch mode (masks_<img_stem>).",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default="jpg,jpeg",
        help="Comma-separated list of image extensions to include in batch mode (case-insensitive).",
    )
    parser.add_argument(
        "--skip_missing_masks",
        action="store_true",
        help="(Batch mode) Skip images whose computed masks directory does not exist.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="(Batch mode) Skip images whose masks directory already contains layout.json.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="(Batch mode) Print what would run but do not execute.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="checkpoints/hf/pipeline.yaml",
        help="Path to the SAM 3D Objects inference pipeline YAML config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for SAM 3D Objects.",
    )
    parser.add_argument(
        "--with_mesh_postprocess",
        type=str2bool,
        default=True,
        help="Whether to use mesh postprocessing.",
    )
    parser.add_argument(
        "--with_texture_baking",
        type=str2bool,
        default=True,
        help="Whether to use texture baking.",
    )
    parser.add_argument(
        "--use_vertex_color",
        type=str2bool,
        default=False,
        help="Whether to use vertex color.",
    )
    parser.add_argument(
        "--with_hand_mesh",
        action="store_true",
        help="Enable mode to process with a provided hand mesh.",
    )
    parser.add_argument(
        "--mask_files",
        type=str,
        nargs="+",
        default=None,
        help="List of mask file names (used with --with_hand_mesh). Full paths are constructed by joining --masks_dir with each name.",
    )
    parser.add_argument(
        "--hand_mask_path",
        type=str,
        default=None,
        help="Path to the hand mask PNG file (used with --with_hand_mesh).",
    )
    parser.add_argument(
        "--hand_mesh_path",
        type=str,
        default=None,
        help="Path to the hand mesh OBJ file (used with --with_hand_mesh).",
    )
    parser.add_argument(
        "--force_layout_post_optimization",
        action="store_true",
        help="Enable forced layout post-optimization (uses ICP for rotation alignment and keeps alignment even if IOU is below threshold).",
    )
    parser.add_argument(
        "--disable_shape_icp",
        action="store_true",
        help="Disable shape ICP in post-optimization.",
    )

    return parser.parse_args()


def _as_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().flatten().tolist()]


def _save_layout_json(outputs: List[dict], plane_quat, plane_translation, objects_dir: str, filename: str = "layout.json") -> str:
    """Save per-object layout transforms (local->scene) to a JSON file."""
    layout_path = os.path.join(objects_dir, filename)
    objects = []
    for i, out in enumerate(outputs):
        rotation = out.get("rotation", None)
        translation = out.get("translation", None)
        scale = out.get("scale", None)
        glb = out.get("glb", None)
        object_name = out.get("object_name", None)
        new_quat = out.get("new_quat", None)

        if rotation is None or translation is None or scale is None or glb is None:
            continue

        # rotation is a quaternion in PyTorch3D convention: (w, x, y, z)
        q_wxyz = rotation.detach().cpu().flatten()
        q_xyzw = torch.stack([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

        R = quaternion_to_matrix(rotation.detach())
        tfm = compose_transform(scale=scale.detach(), rotation=R, translation=translation.detach())
        M = tfm.get_matrix().detach().cpu().squeeze(0)  # (4,4)

        objects.append(
            {
                "index": i,
                "mesh_obj": f"{object_name}.obj",
                "local_to_scene": {
                    "translation": _as_list(translation),
                    "scale": _as_list(scale),
                    "quat_wxyz": _as_list(q_wxyz),
                    "new_quat": _as_list(new_quat),
                    "quat_xyzw": _as_list(q_xyzw),
                    "matrix_4x4_row_major": [float(v) for v in M.reshape(-1).tolist()],
                },
            }
        )
    if plane_quat is not None and plane_translation is not None:
        objects.append(
            {
                "index": len(outputs),
                "mesh_obj": f"plane.obj",
                "local_to_scene": {
                    "translation": plane_translation.ravel().tolist(),
                    "new_quat": plane_quat.ravel().tolist(),
                    "scale": [1.0, 1.0, 1.0],
                },
            }
        )

    payload = {
        "frame": "sam3d_scene",
        "note": (
            "Transforms are local->scene as used by SAM3D make_scene/_export_scene_glb. "
            "Scale is typically uniform (same xyz). Absolute metric scale may be arbitrary."
        ),
        "objects": objects,
    }

    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return layout_path


P3D_TO_ISAAC = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T


def _clip_pointmap(pointmap: torch.Tensor, mask: torch.Tensor, clip_pointmap_beyond_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    if clip_pointmap_beyond_scale is None:
        return pointmap
    pointmap_size = (pointmap.shape[1], pointmap.shape[2])
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask_resized = torchvision.transforms.functional.resize(
        mask, pointmap_size,
        interpolation=torchvision.transforms.InterpolationMode.NEAREST
    ).squeeze(0)

    pointmap_flat = pointmap.reshape(3, -1)
    # Get valid points from the mask
    mask_bool = mask_resized.reshape(-1) > 0.5
    mask_points = pointmap_flat[:, mask_bool]
    mask_distance = mask_points.nanmedian(dim=-1).values[-1]
    logger.info(f"mask_distance: {mask_distance}")

    pointmap_clipped_flat = torch.where(
        pointmap_flat[2, ...].abs() > clip_pointmap_beyond_scale * mask_distance,
        torch.full_like(pointmap_flat, float('nan')),
        pointmap_flat
    )
    pointmap_clipped = pointmap_clipped_flat.reshape(pointmap.shape)
    return pointmap_clipped

def compute_pointmap(image, depth_model, dtype=torch.bfloat16, pointmap=None, clip_pointmap_beyond_scale=None):
    loaded_image = (np.array(image) / 255.0).astype(np.float32)
    loaded_image = torch.from_numpy(loaded_image)
    loaded_mask = loaded_image[..., -1]
    loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]

    device = torch.device("cuda")
    if pointmap is None:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = depth_model(loaded_image)
        pointmaps = output["pointmaps"]
        camera_convention_transform = (
            Transform3d()
            .rotate(camera_to_pytorch3d_camera(device=device).rotation)
            .to(device)
        )
        points_tensor = camera_convention_transform.transform_points(pointmaps)
        intrinsics = output.get("intrinsics", None)
    else:
        output = {}
        points_tensor = pointmap.to(device)
        if loaded_image.shape != points_tensor.shape:
            # Interpolate points_tensor to match loaded_image size
            # loaded_image has shape [3, H, W], we need H and W
            points_tensor = torch.nn.functional.interpolate(
                points_tensor.permute(2, 0, 1).unsqueeze(0),
                size=(loaded_image.shape[1], loaded_image.shape[2]),
                mode="nearest",
            ).squeeze(0).permute(1, 2, 0)
        intrinsics = None

    points_tensor = points_tensor.permute(2, 0, 1)
    points_tensor = _clip_pointmap(points_tensor, loaded_mask, clip_pointmap_beyond_scale=clip_pointmap_beyond_scale)
    
    # Prepare the point map tensor
    point_map_tensor = {
        "pointmap": points_tensor,
        "pts_color": loaded_image,
    }

    # If depth model doesn't provide intrinsics, infer them
    if intrinsics is None:
        intrinsics_result = infer_intrinsics_from_pointmap(
            points_tensor.permute(1, 2, 0), device=device
        )
        point_map_tensor["intrinsics"] = intrinsics_result["intrinsics"]

    return point_map_tensor


def _down_sample_img(img_3chw: torch.Tensor):
    # img_3chw: (3, H, W)
    x = img_3chw.unsqueeze(0)
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    max_side = max(x.shape[2], x.shape[3])
    scale_factor = 1.0

    # heuristics
    if max_side > 3800:
        scale_factor = 0.125
    if max_side > 1900:
        scale_factor = 0.25
    elif max_side > 1200:
        scale_factor = 0.5

    x = torch.nn.functional.interpolate(
        x,
        scale_factor=(scale_factor, scale_factor),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )  # -> (1, 3, H/4, W/4)
    return x.squeeze(0)

def _normalize(v):
    return v / np.linalg.norm(v)

def process_image_with_provided_object(
    image_path,
    masks_path,
    mask_path_for_object,
    inference: Inference,
    mesh_path,
    output_dir: str,
    provided_object_name: str = "provided_object",
    force_layout_post_optimization: bool = False,
    disable_shape_icp: bool = False,
):
    """
    Process an image with multiple masks and one provided object.
    Saves meshes and layout in the same format as _process_one.

    Args:
        image_path: Path to the input image
        masks_path: List of paths to mask files for objects to generate
        mask_path_for_object: Path to mask for the provided object
        inference: Inference pipeline
        mesh_path: Path to directory containing right_vertices.npy and right_faces.npy
        output_dir: Directory to save outputs (meshes and layout.json)
        provided_object_name: Name for the provided object (default: "provided_object")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image = load_image(image_path)
    mask_list = []
    mask_names = []
    for mask_path in masks_path:
        mask = load_mask(mask_path)
        mask_list.append(mask)
        object_name = os.path.splitext(os.path.basename(mask_path))[0]
        mask_names.append(object_name)
    
    mesh = trimesh.load(mesh_path)
    provided_object = {
        "mesh": mesh,
        "mask": load_mask(mask_path_for_object),
        "use_network_pose": False,
        "initial_pose": {
            "rotation": [1, 0, 0, 0],
            "translation": [0, 0, 0], 
            "scale": [1, 1, 1],
        },
    }
    
    print(f"[SAM3D] Processing {len(mask_list)} masks + 1 provided object...")
    result = inference.run_with_provided_object(image, mask_list, provided_object, force_layout_post_optimization=force_layout_post_optimization, disable_shape_icp=disable_shape_icp)
    
    outputs = []
    
    # Process and save generated objects
    for idx, gen_obj in enumerate(result["generated_objects"]):
        object_name = mask_names[idx]
        print(f"Saving generated object {idx + 1}/{len(result['generated_objects'])}: {object_name}")
        
        object_folder = os.path.join(output_dir, object_name)
        os.makedirs(object_folder, exist_ok=True)
        
        glb = gen_obj.get("glb", None)
        if glb is not None:
            output_path = os.path.join(object_folder, f"{object_name}.obj")
            glb.export(output_path)
            print(f"Saved mesh to: {output_path}")
            
            mesh_old = glb.copy()
            mesh_old_vertices = np.zeros_like(mesh_old.vertices)
            for j in range(mesh_old.vertices.shape[0]):
                mesh_old_vertices[j] = mesh_old.vertices[j]
            
            vertices_arr = glb.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
            vertices_tensor = torch.from_numpy(vertices_arr).float().to(gen_obj["rotation"].device)
            R_l2c = quaternion_to_matrix(gen_obj["rotation"])
            l2c_transform = compose_transform(
                scale=gen_obj["scale"],
                rotation=R_l2c,
                translation=gen_obj["translation"],
            )
            transformed_vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
            new_mesh_vertices = transformed_vertices.squeeze(0).cpu().numpy() @ P3D_TO_ISAAC
            glb.vertices = new_mesh_vertices
            
            matrix, transformed, cost = trimesh.registration.procrustes(
                mesh_old_vertices,
                new_mesh_vertices,
                reflection=False,
                return_cost=True
            )
            gen_obj["new_quat"] = torch.tensor(R.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True))
        
        gen_obj["object_name"] = object_name
        gen_obj["glb"] = glb
        outputs.append(gen_obj)
    
    provided_result = result.get("provided_object_result", None)
    if provided_result is not None:
        print(f"  - Saving provided object: {provided_object_name}")
        
        object_folder = os.path.join(output_dir, provided_object_name)
        os.makedirs(object_folder, exist_ok=True)
        
        provided_mesh = provided_result.get("mesh", None)
        if provided_mesh is not None:
            output_path = os.path.join(object_folder, f"{provided_object_name}.obj")
            provided_mesh.export(output_path)
            print(f"    Saved mesh to: {output_path}")
            
            rotation = provided_result.get("rotation", None)
            translation = provided_result.get("translation", None)
            scale = provided_result.get("scale", None)
            
            if rotation is not None and translation is not None and scale is not None:
                print("rotation: ", rotation)
                print("translation: ", translation)
                print("scale: ", scale)
                mesh_old = provided_mesh.copy()
                mesh_old_vertices = np.zeros_like(mesh_old.vertices)
                for j in range(mesh_old.vertices.shape[0]):
                    mesh_old_vertices[j] = mesh_old.vertices[j]
                
                vertices_arr = provided_mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
                vertices_tensor = torch.from_numpy(vertices_arr).float().to(rotation.device)
                R_l2c = quaternion_to_matrix(rotation)
                l2c_transform = compose_transform(
                    scale=scale,
                    rotation=R_l2c,
                    translation=translation,
                )
                transformed_vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
                new_mesh_vertices = transformed_vertices.squeeze(0).cpu().numpy() @ P3D_TO_ISAAC
                provided_mesh.vertices = new_mesh_vertices
                
                matrix, transformed, cost = trimesh.registration.procrustes(
                    mesh_old_vertices,
                    new_mesh_vertices,
                    reflection=False,
                    return_cost=True
                )
                provided_result["new_quat"] = torch.tensor(R.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True))
            
            provided_result["object_name"] = provided_object_name
            provided_result["glb"] = provided_mesh
            outputs.append(provided_result)
    
    _save_layout_json(outputs, None, None, output_dir)
    print(f"[SAM3D] Saved layout.json to: {output_dir}")
    
    print("[SAM3D] Done.")
    return result

def _process_one(
    image_path: str,
    masks_dir: str,
    *,
    inference: Inference,
    depth_model,
    cfg,
    seed: int,
    with_mesh_postprocess: bool = True,
    with_texture_baking: bool = True,
    use_vertex_color: bool = False,
    force_layout_post_optimization: bool = False,
    disable_shape_icp: bool = False,
) -> None:
    mask_paths = [
        os.path.join(masks_dir, f)
        for f in os.listdir(masks_dir)
        if f.lower().endswith(".png")
    ]
    plane_exists = any(os.path.basename(p) == "plane_0.png" for p in mask_paths)
    # if not plane_exists:
    #     print(f"[SAM3D] No plane mask found in: {masks_dir}. Skipping...")
    #     return
    print(f"[SAM3D] Using {len(mask_paths)} masks from: {masks_dir}")

    print(f"[SAM3D] Loading image from: {image_path}")
    image = load_image(image_path)

    print("[SAM3D] Running 3D reconstruction for each object mask (this may take a while)...")
    outputs = []
    
    for i, mask_path in enumerate(mask_paths):
        # if "plane_0.png" in mask_path:
        #     continue
        mask = load_mask(mask_path)
        print(f"  - Reconstructing object {i + 1}/{len(mask_paths)}")
        # Use different seeds per object to decorrelate sampling slightly
        output = inference(image, mask, seed=seed + i, with_mesh_postprocess=with_mesh_postprocess, with_texture_baking=with_texture_baking, use_vertex_color=use_vertex_color, force_layout_post_optimization=force_layout_post_optimization, disable_shape_icp=disable_shape_icp)
        # Include the object name in the output dictionary
        object_name = os.path.splitext(os.path.basename(mask_path))[0]
        output["object_name"] = object_name

        object_folder = os.path.join(masks_dir, object_name)
        os.makedirs(object_folder, exist_ok=True)
        output_path = os.path.join(object_folder, f"{object_name}.obj")
        
        mesh = output["glb"]
        mesh.export(output_path)
        print(f"Saved mesh to: {output_path}")
        
        mesh_old = mesh.copy()
        mesh_old_vertices = np.zeros_like(mesh_old.vertices)
        for j in range(mesh_old.vertices.shape[0]):
            mesh_old_vertices[j] = mesh_old.vertices[j]

        vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
        vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)
        R_l2c = quaternion_to_matrix(output["rotation"])
        l2c_transform = compose_transform(
            scale=output["scale"],
            rotation=R_l2c,
            translation=output["translation"],
        )
        vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
        new_mesh_vertices = vertices.squeeze(0).cpu().numpy() @ P3D_TO_ISAAC
        mesh.vertices = new_mesh_vertices
        matrix, transformed, cost = trimesh.registration.procrustes(
            mesh_old_vertices, 
            new_mesh_vertices,
            reflection=False,
            return_cost=True
        )
        output["new_quat"] = torch.tensor(R.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True))
        outputs.append(output)
        
    # Getting plane mesh

    # rgb_pil = Image.open(image_path).convert("RGB")
    # mask_pil = Image.open(masks_dir + "/plane_0.png")

    # image = np.array(rgb_pil)
    # mask = np.array(mask_pil)

    # if mask.ndim == 2:
    #     mask = mask[..., None]

    # # Replace alpha channel with provided mask
    # image = np.concatenate([image[..., :3], mask], axis=-1)

    # # Match pipeline.yaml defaults
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # dtype_str = str(cfg.get("dtype", "float16"))
    # dtype_map = {
    #     "float16": torch.float16,
    #     "fp16": torch.float16,
    #     "bfloat16": torch.bfloat16,
    #     "bf16": torch.bfloat16,
    #     "float32": torch.float32,
    #     "fp32": torch.float32,
    # }
    # dtype = dtype_map.get(dtype_str.lower(), torch.float16)

    # pointmap_dict = compute_pointmap(
    #     image,
    #     depth_model=depth_model,
    #     dtype=dtype,
    # )
    # # pointmap = pointmap_dict["pointmap"]
    # # intrinsics = pointmap_dict["intrinsics"]

    # floor_mask = _down_sample_img(torch.from_numpy(image[..., -1]).float().unsqueeze(0))[0] > 0.5
    # pts = _down_sample_img(pointmap_dict["pointmap"])
    # pts_hwc = pts.cpu().permute((1, 2, 0))
    # valid_mask_points = floor_mask.cpu().numpy()
    # if valid_mask_points.any():
    #     # Get points within mask
    #     masked_points = pts_hwc[valid_mask_points]
    #     # Filter out invalid points (zero points from depth estimation failures)
    #     valid_points_mask = torch.norm(masked_points, dim=-1) > 1e-6
    #     valid_points = masked_points[valid_points_mask]
    #     points = valid_points.numpy()
    # else:
    #     points = np.array([]).reshape(0, 3)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    # plane_model, inliers = pcd.segment_plane(0.02, 3, 1000)
    # a, b, c, d = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # inlier_points = np.asarray(pcd.points)[inliers]

    # z_range = np.max(inlier_points[:, 2]) - np.min(inlier_points[:, 2])
    # if z_range > 6.0:       # Large range - likely flying points
    #     thresh = 0.90       # Remove 10%
    # elif z_range > 2.0:     # Moderate range
    #     thresh = 0.93       # Remove 7%
    # else:                   # Small range - clean
    #     thresh = 0.95       # Remove 5%
    # depth_quantile = float(np.quantile(inlier_points[:, 2], thresh))
    # clean_points = inlier_points[inlier_points[:, 2] <= depth_quantile]
    # if clean_points.shape[0] < 3:
    #     raise ValueError("Too few points after flying-point removal.")

    # normal = _normalize(np.array([a, b, c], dtype=np.float64))

    # if abs(float(normal[2])) < 0.9:
    #     tangent = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # else:
    #     tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    
    # v1 = _normalize(np.cross(normal, tangent))
    # v2 = _normalize(np.cross(normal, v1))
    # if float(np.dot(np.cross(v1, v2), normal)) < 0.0:
    #     v2 = -v2
    # min_vals = np.min(clean_points, axis=0)
    # max_vals = np.max(clean_points, axis=0)
    # centroid = (min_vals + max_vals) / 2.0 
    # camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # camera_to_centroid = centroid - camera_pos
    # cam_dist = float(np.linalg.norm(camera_to_centroid))
    # if cam_dist < 1e-9:
    #     print("Degenerate: centroid at camera; fallback")
    #     away_axis = v1
    #     perp_axis = v2
    # else:
    #     away_direction = camera_to_centroid / cam_dist
    #     away_in_plane = away_direction - np.dot(away_direction, normal) * normal
    #     away_in_plane_norm = float(np.linalg.norm(away_in_plane))
    #     if away_in_plane_norm > 1e-6:
    #         away_axis = away_in_plane / away_in_plane_norm
    #         perp_axis = _normalize(np.cross(normal, away_axis))
    #     else:
    #         away_axis = v1
    #         perp_axis = v2
    # perp_axis = perp_axis - np.dot(perp_axis, away_axis) * away_axis
    # perp_axis = _normalize(perp_axis)
    # if float(np.dot(np.cross(away_axis, perp_axis), normal)) < 0.0:
    #     perp_axis = -perp_axis

    # # --- Compute extents in camera-aligned (away/perp) coordinates ---
    # relative_points = clean_points - centroid
    # away_coords = np.dot(relative_points, away_axis)
    # perp_coords = np.dot(relative_points, perp_axis)

    # away_min, away_max = np.percentile(away_coords, [0, 100])
    # perp_min, perp_max = np.percentile(perp_coords, [0, 100])

    # away_extent = max(float(away_max - away_min), 0.1)
    # perp_extent = max(float(perp_max - perp_min), 0.1)

    # away_extension_near = away_extent * 0.1
    # away_extension_far = away_extent * 0.5
    # perp_extension = perp_extent * 0.2

    # x_near = away_extent / 2.0 + away_extension_near  # towards camera
    # x_far = away_extent / 2.0 + away_extension_far  # away from camera
    # y_half = perp_extent / 2.0 + perp_extension
    # vertices_local = np.array(
    #     [
    #         [-x_near, -y_half, 0.0],  # 0: near, left
    #         [x_far, -y_half, 0.0],  # 1: far, left
    #         [-x_near, y_half, 0.0],  # 2: near, right
    #         [x_far, y_half, 0.0],  # 3: far, right
    #     ],
    #     dtype=np.float64,
    # )
    # faces = np.array([[0, 1, 3, 2]], dtype=np.int64)
    # canonical_mesh = trimesh.Trimesh(
    #     vertices=vertices_local.astype(np.float32),
    #     faces=faces,
    #     process=False,
    # )
    # canonical_mesh.visual.face_colors = np.array([128, 128, 128, 255], dtype=np.uint8)
    # object_folder = os.path.join(masks_dir, "plane")
    # os.makedirs(object_folder, exist_ok=True)
    # output_path = os.path.join(object_folder, f"plane.obj")
    # canonical_mesh.export(output_path)
    # R_world = np.stack([away_axis, perp_axis, normal], axis=1)  # (3,3)
    # A_world = R_world.T  # (3,3)  (row-vector form)
    # b_world = centroid  # (3,)

    # # vertices_world = vertices_local @ A_world + b_world
    # # vertices_final = vertices_world @ P3D_TO_ISAAC [To be used for confirming that the mesh with baked in transformations is correct]
    # A_final = A_world @ P3D_TO_ISAAC
    # R_final = A_final.T
    # q_wxyz = R.from_matrix(R_final).as_quat(scalar_first=True)
    # translation = b_world @ P3D_TO_ISAAC
    
    _save_layout_json(outputs, None, None, masks_dir)

    print("[SAM3D] Done.")


def main() -> None:
    args = parse_args()
    # Shared config/model init (reused in batch mode)
    print(f"[SAM3D] Loading SAM 3D Objects pipeline from: {args.config}")
    inference = Inference(args.config, compile=False)

    cfg = OmegaConf.load(str(Path(args.config)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_model = instantiate(cfg.depth_model, device=device)

    # Batch mode
    if args.images_dir or args.json_path:
        masks_prefix = str(args.masks_prefix)
        exts = [e.strip() for e in str(args.extensions).split(",") if e.strip()]

        if args.images_dir is not None:
            images_dir = Path(args.images_dir)
            if not images_dir.exists():
                raise FileNotFoundError(f"--images_dir does not exist: {images_dir}")
            if not images_dir.is_dir():
                raise NotADirectoryError(f"--images_dir is not a directory: {images_dir}")
            
            images = list(_iter_images(images_dir, exts))
        else:
            json_path = Path(args.json_path)
            if not json_path.exists():
                raise FileNotFoundError(f"--json_path does not exist: {json_path}")
            if not json_path.is_file():
                raise NotADirectoryError(f"--json_path is not a file: {json_path}")
            
            with open(json_path, "r") as f:
                data = json.load(f)
            images = [Path(item) for item in data]

        # import random
        # num_select = max(1, int(0.1 * len(images)))
        # images = random.sample(images, num_select)
        
        print(f"[SAM3D] Batch mode: found {len(images)} image(s)")
        print(f"[SAM3D] Batch mode: masks_prefix={masks_prefix}")

        for idx, image_path in enumerate(tqdm(images, desc="SAM3D batch", unit="image"), start=1):
            img_stem = image_path.stem
            if args.masks_root is not None:
                masks_root = Path(args.masks_root)
            else:
                masks_root = Path(image_path.parent)
            masks_dir = masks_root / f"{masks_prefix}{img_stem}"

            if not masks_dir.exists():
                msg = f"[SAM3D] ({idx}/{len(images)}) masks dir missing, image={image_path.name} masks_dir={masks_dir}"
                if args.skip_missing_masks:
                    print(msg + " -> SKIP")
                    continue
                raise FileNotFoundError(msg)

            if args.skip_existing and (masks_dir / "layout.json").exists():
                print(f"[SAM3D] ({idx}/{len(images)}) layout.json exists, image={image_path.name} -> SKIP")
                continue

            print(f"[SAM3D] ({idx}/{len(images)}) Processing image={image_path.name} masks_dir={masks_dir}")
            if args.dry_run:
                continue
            _process_one(
                str(image_path),
                str(masks_dir),
                inference=inference,
                depth_model=depth_model,
                cfg=cfg,
                seed=args.seed,
                with_mesh_postprocess=args.with_mesh_postprocess,
                with_texture_baking=args.with_texture_baking,
                use_vertex_color=args.use_vertex_color,
                force_layout_post_optimization=args.force_layout_post_optimization,
                disable_shape_icp=args.disable_shape_icp,
            )

        return

    if args.with_hand_mesh:
        print("Processing image with provided object...")

        # Build full paths from --mask_files names and --masks_dir
        mask_files = [
            os.path.join(args.masks_dir, f)
            for f in args.mask_files
        ]
        print(f"Found {len(mask_files)} mask files in {args.masks_dir}")

        results = process_image_with_provided_object(
            args.image_path,
            mask_files,
            args.hand_mask_path,
            inference,
            args.hand_mesh_path,
            output_dir=args.masks_dir,
            provided_object_name="hand_hawor",
            force_layout_post_optimization=args.force_layout_post_optimization,
            disable_shape_icp=args.disable_shape_icp,
        )
        return
    
    if not args.image_path or not args.masks_dir:
        raise ValueError(
            "Single-image mode requires --image_path and --masks_dir. "
            "Alternatively, use batch mode with --images_dir."
        )

    _process_one(
        args.image_path,
        args.masks_dir,
        inference=inference,
        depth_model=depth_model,
        cfg=cfg,
        seed=args.seed,
        with_mesh_postprocess=args.with_mesh_postprocess,
        with_texture_baking=args.with_texture_baking,
        use_vertex_color=args.use_vertex_color,
        force_layout_post_optimization=args.force_layout_post_optimization,
        disable_shape_icp=args.disable_shape_icp,
    )

if __name__ == "__main__":
    main()