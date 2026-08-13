# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Union, Optional
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from loguru import logger
from PIL import Image

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import Transform3d

from sam3d_objects.model.backbone.dit.embedder.pointmap import PointPatchEmbed
from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.data.dataset.tdfy.transforms_3d import (
    DecomposedTransform,
)
from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
from sam3d_objects.pipeline.inference_utils import o3d_plane_estimation, estimate_plane_area, layout_post_optimization, layout_post_optimization_method_GS


def camera_to_pytorch3d_camera(device="cpu") -> DecomposedTransform:
    """
    R3 camera space --> PyTorch3D camera space
    Also needed for pointmaps
    """
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    return DecomposedTransform(
        rotation=r3_to_p3d_R,
        translation=r3_to_p3d_T,
        scale=torch.tensor(1.0, dtype=r3_to_p3d_R.dtype, device=device),
    )


def recursive_fn_factory(fn):
    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, torch.Tensor):
            return fn(b)
        # Yes, writing out an explicit white list of
        # trivial types is tedious, but so are bugs that
        # come from not applying fn, when expected to have
        # applied it.
        if b is None:
            return b
        trivial_types = [bool, int, float]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None
):
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            f"compiled {fn}" if name is None else name
        ):
            cont_args = recursive_contiguous(args)
            cont_kwargs = recursive_contiguous(kwargs)
            result = compiled_fn(*cont_args, **cont_kwargs)
            cloned_result = recursive_clone(result)
            return cloned_result

    return compiled_fn_wrapper


class InferencePipelinePointMap(InferencePipeline):

    def __init__(
        self, *args, depth_model, layout_post_optimization_method=layout_post_optimization, layout_post_optimization_method_GS=layout_post_optimization_method_GS, clip_pointmap_beyond_scale=None, **kwargs
    ):
        self.depth_model = depth_model
        self.layout_post_optimization_method = layout_post_optimization_method
        self.layout_post_optimization_method_GS = layout_post_optimization_method_GS
        self.clip_pointmap_beyond_scale = clip_pointmap_beyond_scale
        super().__init__(*args, **kwargs)

    def _compile(self):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        compile_mode = "max-autotune"

        for embedder, _ in self.condition_embedders[
            "ss_condition_embedder"
        ].embedder_list:
            if isinstance(embedder, PointPatchEmbed):
                logger.info("Found PointPatchEmbed")
                embedder.inner_forward = compile_wrapper(
                    embedder.inner_forward,
                    mode=compile_mode,
                    fullgraph=True,
                )
            else:
                embedder.forward = compile_wrapper(
                    embedder.forward,
                    mode=compile_mode,
                    fullgraph=True,
                )

        self.models["ss_generator"].reverse_fn.inner_forward = compile_wrapper(
            self.models["ss_generator"].reverse_fn.inner_forward,
            mode=compile_mode,
            fullgraph=True,
        )

        self.models["ss_decoder"].forward = compile_wrapper(
            self.models["ss_decoder"].forward,
            mode=compile_mode,
            fullgraph=True,
        )

        self._warmup()

    def _warmup(self, num_warmup_iters=3):
        test_image = np.ones((512, 512, 4), dtype=np.uint8) * 255
        test_image[:, :, :3] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(test_image)
        mask = None
        image = self.merge_image_and_mask(image, mask)
        with torch.inference_mode(False):
            with torch.no_grad():
                for _ in tqdm(range(num_warmup_iters)):
                    pointmap_dict = recursive_clone(self.compute_pointmap(image))
                    pointmap = pointmap_dict["pointmap"]

                    ss_input_dict = self.preprocess_image(
                        image, self.ss_preprocessor, pointmap=pointmap
                    )
                    ss_return_dict = self.sample_sparse_structure(
                        ss_input_dict, inference_steps=None
                    )

                    _ = self.run_layout_model(
                        ss_input_dict,
                        ss_return_dict,
                        inference_steps=None,
                    )

    def _preprocess_image_and_mask_pointmap(
        self, rgb_image, mask_image, pointmap, img_mask_pointmap_joint_transform
    ):
        for trans in img_mask_pointmap_joint_transform:
            rgb_image, mask_image, pointmap = trans(
                rgb_image, mask_image, pointmap=pointmap
            )
        return rgb_image, mask_image, pointmap

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
        preprocessor,
        pointmap=None,
    ) -> torch.Tensor:
        # canonical type is numpy
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        assert image.ndim == 3  # no batch dimension as of now
        assert image.shape[-1] == 4  # rgba format
        assert image.dtype == np.uint8  # [0,255] range

        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
            rgb_image, rgb_image_mask, pointmap
        )
        
        # Put in a for loop?
        _item = preprocessor_return_dict
        item = {
            "mask": _item["mask"][None].to(self.device),
            "image": _item["image"][None].to(self.device),
            "rgb_image": _item["rgb_image"][None].to(self.device),
            "rgb_image_mask": _item["rgb_image_mask"][None].to(self.device),
        }

        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = _item["pointmap"][None].to(self.device)
            item["rgb_pointmap"] = _item["rgb_pointmap"][None].to(self.device)
            item["pointmap_scale"] = _item["pointmap_scale"][None].to(self.device)
            item["pointmap_shift"] = _item["pointmap_shift"][None].to(self.device)
            item["rgb_pointmap_scale"] = _item["rgb_pointmap_scale"][None].to(self.device)
            item["rgb_pointmap_shift"] = _item["rgb_pointmap_shift"][None].to(self.device)

        # Add unnormed pointmap for post-optimization
        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            full_pointmap = self._apply_transform(
                pointmap, preprocessor.pointmap_transform
            )
            item["rgb_pointmap_unnorm"] = full_pointmap[None].to(self.device)            

        return item

    def _clip_pointmap(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.clip_pointmap_beyond_scale is None:
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
            pointmap_flat[2, ...].abs() > self.clip_pointmap_beyond_scale * mask_distance,
            torch.full_like(pointmap_flat, float('nan')),
            pointmap_flat
        )
        pointmap_clipped = pointmap_clipped_flat.reshape(pointmap.shape)
        return pointmap_clipped

    def refine_scale(self, revised_scale):
        # Check if all three channels of revised_scale are close to each other
        if not torch.allclose(revised_scale[0, 0:1], revised_scale[0, 1:2], atol=1e-3) or \
           not torch.allclose(revised_scale[0, 0:1], revised_scale[0, 2:3], atol=1e-3):
            logger.warning(
                f"revised_scale values are not close (tolerance=1e-3): "
            )
        # Use 3-channel mean value
        revised_scale = revised_scale.clone()
        mean_val = revised_scale.mean(dim=1, keepdim=True)
        revised_scale[:] = mean_val

        return revised_scale

    def compute_pointmap(self, image, pointmap=None):
        loaded_image = self.image_to_float(image)
        loaded_image = torch.from_numpy(loaded_image)
        loaded_mask = loaded_image[..., -1]
        loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]

        if pointmap is None:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    output = self.depth_model(loaded_image)
            pointmaps = output["pointmaps"]
            camera_convention_transform = (
                Transform3d()
                .rotate(camera_to_pytorch3d_camera(device=self.device).rotation)
                .to(self.device)
            )
            points_tensor = camera_convention_transform.transform_points(pointmaps)
            intrinsics = output.get("intrinsics", None)
        else:
            output = {}
            points_tensor = pointmap.to(self.device)
            if loaded_image.shape != points_tensor.shape:
                # Interpolate points_tensor to match loaded_image size
                # loaded_image has shape [3, H, W], we need H and W
                points_tensor = torch.nn.functional.interpolate(
                    points_tensor.permute(2, 0, 1).unsqueeze(0),
                    size=(loaded_image.shape[1], loaded_image.shape[2]),
                    mode="nearest",
                ).squeeze(0).permute(1, 2, 0)
            intrinsics = None
        
        # Prepare the point map tensor
        point_map_tensor = {
            "pts_color": loaded_image,
        }

        # If depth model doesn't provide intrinsics, infer them
        if intrinsics is None:
            camera_convention_transform = (
                Transform3d()
                .rotate(camera_to_pytorch3d_camera(device=self.device).rotation)
                .to(self.device)
            )
            points_tensor_moge = camera_convention_transform.inverse().transform_points(points_tensor)
            intrinsics_result = infer_intrinsics_from_pointmap(
                points_tensor_moge, device=self.device
            )
            point_map_tensor["intrinsics"] = intrinsics_result["intrinsics"]
        else:
            point_map_tensor["intrinsics"] = intrinsics

        points_tensor = points_tensor.permute(2, 0, 1)
        points_tensor = self._clip_pointmap(points_tensor, loaded_mask)
        point_map_tensor["pointmap"] = points_tensor

        return point_map_tensor

    @torch.autograd.grad_mode.inference_mode(mode=False)
    def run_post_optimization(self, mesh_glb, intrinsics, pose_dict, layout_input_dict, fill_mask_holes=False, force_alignment=False, fixed_scale=None, Enable_visible_ICP=False, Enable_shape_ICP=True):
        intrinsics = intrinsics.clone()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        re_focal = min(fx, fy)
        intrinsics[0, 0], intrinsics[1, 1] = re_focal, re_focal
        
        # Convert pose_dict values to tensors if they are lists
        rotation = pose_dict["rotation"]
        translation = pose_dict["translation"]
        scale = pose_dict["scale"]
        
        if isinstance(rotation, list):
            rotation = torch.tensor(rotation, dtype=torch.float32, device=self.device)
            if rotation.dim() == 1:
                rotation = rotation.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
            elif rotation.dim() == 2:
                rotation = rotation.unsqueeze(0)  # [1, N, 4]
        
        if isinstance(translation, list):
            translation = torch.tensor(translation, dtype=torch.float32, device=self.device)
            if translation.dim() == 1:
                translation = translation.unsqueeze(0)  # [1, 3]
        
        if isinstance(scale, list):
            scale = torch.tensor(scale, dtype=torch.float32, device=self.device)
            if scale.dim() == 1:
                scale = scale.unsqueeze(0)  # [1, 3]
        
        # Update pose_dict with tensors
        pose_dict = {
            "rotation": rotation,
            "translation": translation,
            "scale": scale,
        }
        
        mask = layout_input_dict["rgb_image_mask"][0, 0]
        
        # Optionally fill holes in the mask to avoid internal occlusion detection
        if fill_mask_holes:
            from scipy.ndimage import binary_fill_holes
            mask_np = mask.cpu().numpy() > 0.5
            mask_filled = binary_fill_holes(mask_np)
            mask = torch.from_numpy(mask_filled.astype(np.float32)).to(mask.device)
            logger.info("Filled holes in mask for post-optimization")
        
        if force_alignment:
            # Use the force_alignment version that keeps alignment even if IOU is low
            # Enable ICP for rotation alignment when force_alignment is True
            revised_quat, revised_t, revised_scale, final_iou, _, _ = (
                self._layout_post_optimization_force_alignment(
                    mesh_glb,
                    pose_dict["rotation"],
                    pose_dict["translation"],
                    pose_dict["scale"],
                    mask,
                    layout_input_dict["rgb_pointmap_unnorm"][0].permute(1, 2, 0),
                    intrinsics,
                    Enable_shape_ICP=Enable_shape_ICP,
                    Enable_visible_ICP=Enable_visible_ICP,
                    min_size=518,
                    device=self.device,
                    fixed_scale=fixed_scale,
                )
            )
        else:
            revised_quat, revised_t, revised_scale, final_iou, _, _ = (
                self.layout_post_optimization_method(
                    mesh_glb,
                    pose_dict["rotation"],
                    pose_dict["translation"],
                    pose_dict["scale"],
                    mask,
                    layout_input_dict["rgb_pointmap_unnorm"][0].permute(1, 2, 0),
                    intrinsics,
                    Enable_shape_ICP=False,
                    min_size=518,
                    device=self.device,
                )
            )

        revised_scale = self.refine_scale(revised_scale)
        return {
            "rotation": revised_quat,
            "translation": revised_t,
            "scale": revised_scale,
            "iou": final_iou,
        }
    
    @torch.autograd.grad_mode.inference_mode(mode=False)
    def _layout_post_optimization_force_alignment(
        self,
        Mesh,
        quaternion,
        translation,
        scale,
        mask,
        point_map,
        intrinsics,
        Enable_shape_ICP=True,
        Enable_visible_ICP=False,
        Enable_rendering_optimization=True,
        min_size=512,
        device=None,
        fixed_scale=None,
    ):
        """
        Modified version of layout_post_optimization that keeps alignment
        even when render-compare IOU is below threshold.

        If ``fixed_scale`` is provided, step 3 (render-compare) will not
        optimize scale — only rotation and translation are optimized.
        """
        from .layout_post_optimization_utils import (
            get_mesh, get_mask_renderer, check_occlusion, run_alignment,
            run_ICP, run_render_compare, apply_transform, compute_iou, set_seed,
            get_visible_vertices
        )
        from pytorch3d.structures import Meshes
        from pytorch3d.transforms import Transform3d, quaternion_to_matrix, matrix_to_quaternion
        from .inference_utils import compose_transform
        
        logger.info(f"Starting force_alignment post-optimization!")
        set_seed(100)
        if device is None:
            device = quaternion.device

        # Initialize transformation and process mesh
        Rotation = quaternion_to_matrix(quaternion.squeeze(1))
        center = translation[0].clone()
        tfm_ori = compose_transform(scale=scale, rotation=Rotation, translation=translation)
        mesh, faces_idx, textures = get_mesh(Mesh, tfm_ori, device)

        mask, renderer = get_mask_renderer(mask, min_size, intrinsics, device)
        logger.info(f"Loaded!")

        source_points, target_points, center, tfm1, mesh, ori_iou, final_iou, flag_notgt = (
            run_alignment(
                point_map, mask, mesh, center, faces_idx, textures, renderer, device
            )
        )
        logger.info(f"Step 1 (Manual Alignment) Done! ori_iou={ori_iou}, final_iou={final_iou}")

        # Print predicted translation, rotation, and scale after alignment
        tfm_after_align = tfm_ori.compose(tfm1)
        M_after_align = tfm_after_align.get_matrix()[0]
        T_after_align = M_after_align[3, :3]
        A_after_align = M_after_align[:3, :3]
        scale_after_align = A_after_align.norm(dim=1)
        R_after_align = A_after_align / scale_after_align[:, None]
        quat_after_align = matrix_to_quaternion(R_after_align)
        logger.info(f"[After Alignment] Translation: {T_after_align.tolist()}")
        logger.info(f"[After Alignment] Rotation (quaternion): {quat_after_align.tolist()}")
        logger.info(f"[After Alignment] Scale: {scale_after_align.tolist()}")

        if flag_notgt:
            logger.warning("No target points found, returning original pose")
            return (quaternion, translation, scale, -1.0, False, False)

        if Enable_shape_ICP:
            Flag_ICP = True

            # Compute visible-only source points for ICP if enabled
            visible_source_points = None
            if Enable_visible_ICP:
                vis_idx = get_visible_vertices(mesh, renderer, mask)
                if vis_idx.numel() > 0:
                    visible_source_points = mesh.verts_packed()[vis_idx]
                    logger.info(f"Visible ICP: {vis_idx.numel()}/{mesh.verts_packed().shape[0]} vertices")
                else:
                    logger.warning("Visible ICP: no visible vertices, falling back to all")

            points_aligned_icp, transformation = run_ICP(
                mesh, source_points, target_points, threshold=0.05,
                visible_source_points=visible_source_points
            )
            mesh_ICP = Meshes(
                verts=[points_aligned_icp], faces=[faces_idx], textures=textures
            )
            rendered = renderer(mesh_ICP)
            ori_iou_shapeICP = compute_iou(
                rendered[..., 3][0][None, None], mask, threshold=0.5
            )
            # determine whether accept ICP
            if ori_iou_shapeICP > ori_iou:
                mesh = mesh_ICP
                final_iou = ori_iou_shapeICP.cpu().item()
                T_o3d = torch.tensor(transformation, dtype=torch.float32, device=device)
                T_o3d = T_o3d.T
                A = T_o3d[:3, :3]
                t = T_o3d[3, :3]
                scale_icp = A.norm(dim=1)
                R = A / scale_icp[:, None]
                center = ((center[None] * scale_icp) @ R + t)[0]
                tfm2 = (
                    Transform3d(device=device)
                    .scale(scale_icp[None])
                    .rotate(R[None])
                    .translate(t[None])
                )
                logger.info(f"Step 2 (ICP) accepted, iou improved to {final_iou}")
            else:
                Flag_ICP = False
                scale_2, translation_2 = torch.tensor(1).to(device), torch.zeros([3]).to(device)
                tfm2 = (
                    Transform3d(device=device)
                    .scale(scale_2.expand(3)[None])
                    .translate(translation_2[None])
                )
                logger.info(f"Step 2 (ICP) rejected, keeping previous iou")
        else:
            Flag_ICP = False
            scale_2, translation_2 = torch.tensor(1).to(device), torch.zeros([3]).to(device)
            tfm2 = (
                Transform3d(device=device)
                .scale(scale_2.expand(3)[None])
                .translate(translation_2[None])
            )
        logger.info(f"Step 2 Done!")

        # Print predicted translation, rotation, and scale after ICP
        tfm_after_icp = tfm_ori.compose(tfm1).compose(tfm2)
        M_after_icp = tfm_after_icp.get_matrix()[0]
        T_after_icp = M_after_icp[3, :3]
        A_after_icp = M_after_icp[:3, :3]
        scale_after_icp = A_after_icp.norm(dim=1)
        R_after_icp = A_after_icp / scale_after_icp[:, None]
        quat_after_icp = matrix_to_quaternion(R_after_icp)
        if Enable_shape_ICP:
            logger.info(f"[After ICP] Translation: {T_after_icp.tolist()}")
            logger.info(f"[After ICP] Rotation (quaternion): {quat_after_icp.tolist()}")
            logger.info(f"[After ICP] Scale: {scale_after_icp.tolist()}")
        else:
            logger.info("ICP was disabled")

        # Correct mesh scale back to fixed_scale before render-compare,
        # so step 3 optimizes rotation/translation on a correctly-sized mesh.
        if fixed_scale is not None:
            fs = fixed_scale.detach().squeeze().to(device)  # [3]
            ratio = fs / scale_after_icp
            verts = mesh.verts_packed()
            mesh = Meshes(
                verts=[(verts - T_after_icp[None]) * ratio[None] + T_after_icp[None]],
                faces=[faces_idx], textures=textures,
            )
            center = (center - T_after_icp) * ratio + T_after_icp
            logger.info(f"[Scale fix before step 3] {scale_after_icp.tolist()} -> {fs.tolist()}")

        if not Enable_rendering_optimization:
            Flag_optim = False
            tfm = tfm_ori.compose(tfm1).compose(tfm2)
        else:
            quat, translation_opt, scale_opt, R = run_render_compare(
                mesh, center, renderer, mask, device,
                optimize_scale=(fixed_scale is None),
            )
            with torch.no_grad():
                transformed = apply_transform(mesh, center, quat, translation_opt, scale_opt)
                rendered = renderer(transformed)
            optimized_iou = compute_iou(
                rendered[..., 3][0][None, None], mask, threshold=0.5
            )
            iou_before_rc = final_iou  # IoU after alignment + ICP (before render-compare)
            logger.info(f"Step 3 (Render-Compare) iou={optimized_iou.item():.4f}, iou_before_rc={iou_before_rc:.4f}, ori_iou={ori_iou}")

            # Only apply render-compare if it improves IoU
            if optimized_iou.item() >= iou_before_rc:
                Flag_optim = True
                final_iou = optimized_iou.detach().cpu().item()
                tfm3 = (
                    Transform3d(device=device)
                    .translate(-center[None])
                    .scale(scale_opt.expand(3)[None])
                    .rotate(R.T[None])
                    .translate(center[None])
                    .translate(translation_opt[None])
                )
                tfm = tfm_ori.compose(tfm1).compose(tfm2).compose(tfm3)
                logger.info(f"Render-compare accepted, iou improved {iou_before_rc:.4f} -> {final_iou:.4f}")
            else:
                Flag_optim = False
                tfm = tfm_ori.compose(tfm1).compose(tfm2)
                logger.info(f"Render-compare rejected, iou would decrease {iou_before_rc:.4f} -> {optimized_iou.item():.4f}. Keeping pre-RC pose.")

            # Print predicted translation, rotation, and scale after render-compare
            M_after_rc = tfm.get_matrix()[0]
            T_after_rc = M_after_rc[3, :3]
            A_after_rc = M_after_rc[:3, :3]
            scale_after_rc = A_after_rc.norm(dim=1)
            R_after_rc = A_after_rc / scale_after_rc[:, None]
            quat_after_rc = matrix_to_quaternion(R_after_rc)
            logger.info(f"[After Render-Compare] Translation: {T_after_rc.tolist()}")
            logger.info(f"[After Render-Compare] Rotation (quaternion): {quat_after_rc.tolist()}")
            logger.info(f"[After Render-Compare] Scale: {scale_after_rc.tolist()}")
        
        logger.info(f"Step 3 Done!")
        M = tfm.get_matrix()[0]
        T_final = M[3, :3][None]
        A = M[:3, :3]
        scale_final = A.norm(dim=1)[None]
        R_final = A / scale_final[:, None]
        quat_final = matrix_to_quaternion(R_final)

        # Print final predicted translation, rotation, and scale
        logger.info(f"[Final] Translation: {T_final.tolist()}")
        logger.info(f"[Final] Rotation (quaternion): {quat_final.tolist()}")
        logger.info(f"[Final] Scale: {scale_final.tolist()}")

        return (
            quat_final,
            T_final,
            scale_final,
            round(float(final_iou), 4),
            Flag_ICP,
            Flag_optim,
        )

    @torch.autograd.grad_mode.inference_mode(mode=False)
    def run_post_optimization_GS(self, gs_input, intrinsics, pose_dict, layout_input_dict, backend="gsplat"):
        intrinsics = intrinsics.clone()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        re_focal = min(fx, fy)
        intrinsics[0, 0], intrinsics[1, 1] = re_focal, re_focal

        revised_quat, revised_t, revised_scale, final_iou, initial_iou, _, flag_optim = (
            self.layout_post_optimization_method_GS(
                gs_input,
                pose_dict["rotation"],
                pose_dict["translation"],
                pose_dict["scale"],
                layout_input_dict["rgb_image_mask"][0, 0],
                layout_input_dict["rgb_image"][0],
                layout_input_dict["rgb_pointmap_unnorm"][0].permute(1, 2, 0),
                intrinsics,
                Enable_occlusion_check=False,
                Enable_manual_alignment=False,
                Enable_shape_ICP=False,
                Enable_rendering_optimization=True,
                min_size=518,
                device=self.device,
                backend=backend,
            )
        )

        revised_scale = self.refine_scale(revised_scale)
        return {
            "rotation": revised_quat,
            "translation": revised_t,
            "scale": revised_scale,
            "iou": final_iou,
            "iou_before_optim": initial_iou,
            "optim_accepted": flag_optim,
        }

    def run_with_provided_object(
        self,
        rgba_images: list,
        provided_object: dict = None,  # {mesh, rgba_image, initial_pose: {translation, rotation, scale}}
        seed: Optional[int] = None,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=False,
        use_vertex_color=False,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        pointmap=None,
        decode_formats=None,
        force_layout_post_optimization=False,
        disable_shape_icp=False,
    ) -> dict:
        with self.device:
            results = {
                "generated_objects": [],
                "provided_object_result": None,
            }
            
            for idx, rgba_image in enumerate(rgba_images):
                logger.info(f"Processing generated object {idx + 1}/{len(rgba_images)}")
                
                if isinstance(rgba_image, Image.Image):
                    rgba_image = np.array(rgba_image)
                
                image = rgba_image
                pointmap_dict = self.compute_pointmap(image, pointmap)
                object_pointmap = pointmap_dict["pointmap"]
                
                ss_input_dict = self.preprocess_image(
                    image, self.ss_preprocessor, pointmap=object_pointmap
                )
                slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
                
                if seed is not None:
                    torch.manual_seed(seed + idx)  # Different seed per object
                
                ss_return_dict = self.sample_sparse_structure(
                    ss_input_dict,
                    inference_steps=stage1_inference_steps,
                    use_distillation=use_stage1_distillation,
                )
                
                pointmap_scale = ss_input_dict.get("pointmap_scale", None)
                pointmap_shift = ss_input_dict.get("pointmap_shift", None)
                ss_return_dict.update(
                    self.pose_decoder(
                        ss_return_dict,
                        scene_scale=pointmap_scale,
                        scene_shift=pointmap_shift,
                    )
                )
                initial_pose_dict = {
                    "rotation": ss_return_dict["rotation"],
                    "translation": ss_return_dict["translation"],
                    "scale": ss_return_dict["scale"],
                }
                
                logger.info(f"Rescaling scale by {ss_return_dict['downsample_factor']} after downsampling")
                ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]
                
                if stage1_only:
                    ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                    pts = type(self)._down_sample_img(object_pointmap)
                    pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])
                    results["generated_objects"].append({
                        **ss_return_dict,
                        "object_idx": idx,
                        "pointmap": pts.cpu().permute((1, 2, 0)),
                        "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),
                        "intrinsics": pointmap_dict.get("intrinsics", None),
                    })
                    continue
                
                # Stage 2: Sample structured latent and decode
                coords = ss_return_dict["coords"]
                slat = self.sample_slat(
                    slat_input_dict,
                    coords,
                    inference_steps=stage2_inference_steps,
                    use_distillation=use_stage2_distillation,
                )
                
                outputs = self.decode_slat(
                    slat, self.decode_formats if decode_formats is None else decode_formats
                )
                outputs = self.postprocess_slat_output(
                    outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
                )
                
                glb = outputs.get("glb", None)
                gs_input = None  # outputs.get("gaussian", None)
                
                # Optional layout post-optimization
                try:
                    if with_layout_postprocess:
                        if (
                            gs_input is not None
                            and self.layout_post_optimization_method_GS is not None
                        ):
                            logger.info("Running GS layout post optimization for generated object...")
                            postprocessed_pose = self.run_post_optimization_GS(
                                deepcopy(gs_input[0]),
                                pointmap_dict["intrinsics"],
                                ss_return_dict,
                                ss_input_dict,
                                backend="gsplat",
                            )
                            ss_return_dict.update(postprocessed_pose)
                        elif (
                            glb is not None
                            and self.layout_post_optimization_method is not None
                        ):
                            logger.info("Running mesh layout post optimization for generated object...")
                            postprocessed_pose = self.run_post_optimization(
                                deepcopy(glb),
                                pointmap_dict["intrinsics"],
                                ss_return_dict,
                                ss_input_dict,
                                force_alignment=force_layout_post_optimization,
                                Enable_shape_ICP=not disable_shape_icp,
                            )
                            ss_return_dict.update(postprocessed_pose)
                except Exception as e:
                    logger.error(
                        f"Error during layout post optimization for object {idx}: {e}",
                        exc_info=True,
                    )
                
                pts = type(self)._down_sample_img(object_pointmap)
                pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])
                results["generated_objects"].append({
                    **ss_return_dict,
                    **outputs,
                    "object_idx": idx,
                    "pointmap": pts.cpu().permute((1, 2, 0)),
                    "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),
                    "intrinsics": pointmap_dict.get("intrinsics", None),
                    "initial_pose": initial_pose_dict,
                    "refined_pose": ss_return_dict,
                })
            
            
            if provided_object is not None:
                logger.info("Processing provided object...")
                
                provided_mesh = provided_object["mesh"]
                provided_rgba_image = provided_object["rgba_image"] 
                initial_pose = provided_object.get("initial_pose", None)
                
                if isinstance(provided_rgba_image, Image.Image):
                    provided_rgba_image = np.array(provided_rgba_image)
                
                provided_pointmap_dict = self.compute_pointmap(provided_rgba_image, pointmap)
                provided_object_pointmap = provided_pointmap_dict["pointmap"]
                
                provided_ss_input_dict = self.preprocess_image(
                    provided_rgba_image, self.ss_preprocessor, pointmap=provided_object_pointmap
                )
                
                pointmap_scale = provided_ss_input_dict.get("pointmap_scale", None)
                pointmap_shift = provided_ss_input_dict.get("pointmap_shift", None)
                

                logger.info("Using provided initial pose for provided object...")
                
                if initial_pose is None:
                    raise ValueError(
                        "initial_pose must be provided when use_network_pose=False"
                    )

                # Assuming pose is already in Y-UP/pipeline coordinates
                provided_pose_dict = {
                    "rotation": initial_pose["rotation"],
                    "translation": initial_pose["translation"],
                    "scale": initial_pose["scale"],
                }
                
                # Store pose before refinement
                initial_pose_result = deepcopy(provided_pose_dict)
                
                # Run layout post-optimization to refine the pose
                refined_pose = None
                refinement_skipped = False
                try:
                    if self.layout_post_optimization_method is not None and with_layout_postprocess:
                        logger.info("Running mesh layout post optimization for provided object...")
                        logger.info(f"  Input pose - rotation: {provided_pose_dict['rotation']}, translation: {provided_pose_dict['translation']}, scale: {provided_pose_dict['scale']}")
                        
                        refined_pose = self.run_post_optimization(
                            deepcopy(provided_mesh),
                            provided_pointmap_dict["intrinsics"],
                            provided_pose_dict,
                            provided_ss_input_dict,
                            fill_mask_holes=False,  # Fill holes to avoid internal occlusion detection (e.g., for hands)
                            force_alignment=force_layout_post_optimization,
                            Enable_shape_ICP=not disable_shape_icp,
                        )
                        
                        logger.info(f"  Output pose - rotation: {refined_pose['rotation']}, translation: {refined_pose['translation']}, scale: {refined_pose['scale']}")
                        
                        if refined_pose.get("iou", 0) < 0:
                            logger.warning("Post-optimization was skipped (likely due to occlusion or no valid target points)")
                            refinement_skipped = True
                        else:
                            logger.info(f"Refinement IOU: {refined_pose.get('iou', 'N/A')}")
                            provided_pose_dict.update(refined_pose)
                            logger.info("Finished pose refinement for provided object!")
                except Exception as e:
                    logger.error(
                        f"Error during pose refinement for provided object: {e}",
                        exc_info=True,
                    )
                
                pts = type(self)._down_sample_img(provided_object_pointmap)
                pts_colors = type(self)._down_sample_img(provided_pointmap_dict["pts_color"])
                results["provided_object_result"] = {
                    "mesh": provided_mesh,
                    "initial_pose": initial_pose_result,
                    "refined_pose": refined_pose if not refinement_skipped else None,
                    "refinement_skipped": refinement_skipped,
                    "refinement_iou": refined_pose.get("iou", None) if refined_pose else None,
                    "final_pose": provided_pose_dict,
                    "pointmap": pts.cpu().permute((1, 2, 0)),
                    "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),
                    "intrinsics": provided_pointmap_dict.get("intrinsics", None),
                    **provided_pose_dict,  # rotation, translation, scale at top level
                }
            
            logger.info("Finished processing all objects!")
            return results
        
    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed: Optional[int] = None,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=False,
        use_vertex_color=False,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        pointmap=None,
        decode_formats=None,
        estimate_plane=False,
        force_layout_post_optimization=False,
        disable_shape_icp=False,
    ) -> dict:
        image = self.merge_image_and_mask(image, mask)
        with self.device: 
            pointmap_dict = self.compute_pointmap(image, pointmap)
            pointmap = pointmap_dict["pointmap"]
            pts = type(self)._down_sample_img(pointmap)
            pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])

            if estimate_plane:
                return self.estimate_plane(pointmap_dict, image)

            ss_input_dict = self.preprocess_image(
                image, self.ss_preprocessor, pointmap=pointmap
            )

            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            if seed is not None:
                torch.manual_seed(seed)
            ss_return_dict = self.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
            )

            # We could probably use the decoder from the models themselves
            pointmap_scale = ss_input_dict.get("pointmap_scale", None)
            pointmap_shift = ss_input_dict.get("pointmap_shift", None)
            ss_return_dict.update(
                self.pose_decoder(
                    ss_return_dict,
                    scene_scale=pointmap_scale,
                    scene_shift=pointmap_shift,
                )
            )

            logger.info(f"Rescaling scale by {ss_return_dict['downsample_factor']} after downsampling")
            ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]

            if stage1_only:
                logger.info("Finished!")
                ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                return {
                    **ss_return_dict,
                    "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                    "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),  # HxWx3
                }
                # return ss_return_dict

            coords = ss_return_dict["coords"]
            slat = self.sample_slat(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
            )
            outputs = self.decode_slat(
                slat, self.decode_formats if decode_formats is None else decode_formats
            )
            outputs = self.postprocess_slat_output(
                outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
            )
            glb = outputs.get("glb", None)
            #gs_input = outputs.get("gaussian", None)
            gs_input = None
            try:
                if with_layout_postprocess:
                    if (
                        gs_input is not None
                        and self.layout_post_optimization_method_GS is not None
                    ):
                        logger.info("Running GS layout post optimization method...")
                        postprocessed_pose = self.run_post_optimization_GS(
                            deepcopy(gs_input[0]),
                            pointmap_dict["intrinsics"],
                            ss_return_dict,
                            ss_input_dict,
                            backend="gsplat",
                        )
                        ss_return_dict.update(postprocessed_pose)
                        logger.info(f"Finished GS post-optimization!")
                    elif (
                        glb is not None
                        and self.layout_post_optimization_method is not None
                    ):
                        logger.info("Running mesh layout post optimization method...")
                        postprocessed_pose = self.run_post_optimization(
                            deepcopy(glb),
                            pointmap_dict["intrinsics"],
                            ss_return_dict,
                            ss_input_dict,
                            force_alignment=force_layout_post_optimization,
                            Enable_shape_ICP=not disable_shape_icp,
                        )
                        ss_return_dict.update(postprocessed_pose)
                        logger.info("Finished mesh post-optimization!")
                    else:
                        logger.info("No post-optimization method available (no GS or mesh found)")
            except Exception as e:
                logger.error(
                    f"Error during layout post optimization: {e}", exc_info=True
                )

            logger.info("Finished!")

            return {
                **ss_return_dict,
                **outputs,
                "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),  # HxWx3
            }

    @staticmethod
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

    def estimate_plane(self, pointmap_dict, image, ground_area_threshold=0.25, min_points=100):
        assert image.shape[-1] == 4  # rgba format
        # Extract mask from alpha channel
        floor_mask = type(self)._down_sample_img(torch.from_numpy(image[..., -1]).float().unsqueeze(0))[0] > 0.5
        pts = type(self)._down_sample_img(pointmap_dict["pointmap"])

        # Get all points in 3D space (H, W, 3)
        pts_hwc = pts.cpu().permute((1, 2, 0))

        valid_mask_points = floor_mask.cpu().numpy()
        # Extract points that fall within the mask
        if valid_mask_points.any():
            # Get points within mask
            masked_points = pts_hwc[valid_mask_points]
            # Filter out invalid points (zero points from depth estimation failures)
            valid_points_mask = torch.norm(masked_points, dim=-1) > 1e-6
            valid_points = masked_points[valid_points_mask]
            points = valid_points.numpy()
        else:
            points = np.array([]).reshape(0, 3)
     
        # Calculate area coverage and check num of points
        overlap_area = estimate_plane_area(floor_mask)
        has_enough_points = len(points) >= min_points

        logger.info(f"Plane estimation: {len(points)} points, {overlap_area:.3f} area coverage")
        if overlap_area > ground_area_threshold and has_enough_points:
            try:
                mesh = o3d_plane_estimation(points)
                logger.info("Successfully estimated plane mesh")
            except Exception as e:
                logger.error(f"Failed to estimate plane: {e}")
                mesh = None
        else:
            logger.info(f"Skipping plane estimation: area={overlap_area:.3f}, points={len(points)}")
            mesh = None

        return {
            "glb": mesh,
            "translation": torch.tensor([[0.0, 0.0, 0.0]]),
            "scale": torch.tensor([[1.0, 1.0, 1.0]]),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        }
