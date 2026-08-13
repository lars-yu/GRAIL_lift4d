from enum import Enum

import numpy as np
import torch
from pytorch3d.renderer import (
    AmbientLights,
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    MeshRendererWithFragments,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
)
from pytorch3d.renderer.mesh.shader import HardDepthShader, SoftDepthShader
from pytorch3d.structures import Meshes


class RendererType(Enum):
    SOFT_PHONG = "soft_phong"
    SOFT_SILHOUETTE = "soft_silhouette"
    HARD_DEPTH = "hard_depth"
    HARD_SILHOUETTE = "hard_silhouette"
    HARD_PHONG = "hard_phong"


def create_raster_settings(
    image_size, blur_radius=0.0, faces_per_pixel=1, bin_size=0, max_faces_per_bin=50000
):
    """
    Create rasterization settings for PyTorch3D renderer

    Args:
        image_size (tuple): (height, width) of output images
        blur_radius (float): Blur radius for rasterization
        faces_per_pixel (int): Maximum faces per pixel
        bin_size (int): Bin size for rasterization (0 = naive rasterization)
        max_faces_per_bin (int): Maximum faces per bin

    Returns:
        RasterizationSettings: Configured rasterization settings
    """
    return RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
    )


def create_lights(device="cuda", location=None):
    """
    Create lighting for PyTorch3D renderer

    Args:
        device (str): Device for computations
        location (list): Light position [x, y, z]

    Returns:
        PointLights: Configured lighting
    """
    if location is None:
        location = [[0.0, 0.0, -3.0]]  # Light in front of camera

    return PointLights(device=device, location=location)


def create_neutral_lights(device="cuda"):
    """
    Create neutral ambient lighting that preserves original vertex colors
    without directional lighting effects.

    Args:
        device (str): Device for computations

    Returns:
        AmbientLights: Ambient lighting that preserves vertex colors
    """
    return AmbientLights(device=device)


def create_camera_relative_lights(cameras, device="cuda", offset_cam=(0.0, 0.0, -3.0)):
    """
    Create point lights located at a fixed offset in the camera coordinate frame,
    transformed into world coordinates based on the camera pose.

    Args:
        cameras: PyTorch3D cameras object (batch supported)
        device (str): Device for computations
        offset_cam (tuple|list): Light position in camera coordinates (x, y, z)

    Returns:
        PointLights: Point lights positioned relative to camera pose
    """
    # Determine batch size from cameras (default to 1 if attributes are absent)
    try:
        batch_size = int(cameras.R.shape[0])
    except Exception:
        batch_size = 1

    # Offset in camera space replicated for each camera in the batch
    offset_cam_tensor = torch.tensor(offset_cam, dtype=torch.float32, device=device).view(1, 1, 3)
    offsets_cam = offset_cam_tensor.repeat(batch_size, 1, 1)  # (N, 1, 3)

    # Transform to world space using the inverse of world-to-view transform
    cam_to_world = cameras.get_world_to_view_transform().inverse()
    offsets_world = cam_to_world.transform_points(offsets_cam).squeeze(1)  # (N, 3)

    return PointLights(device=device, location=offsets_world)


def create_renderer(
    cameras,
    image_size,
    renderer_type=RendererType.SOFT_PHONG,
    device="cuda",
    neutral_light=False,
    background_color=None,
):
    """
    Set up a default renderer for GRAIL rendering

    Args:
        cameras: PyTorch3D cameras
        image_size (tuple): (height, width) of output images
        renderer_type: Type of renderer to create
        device (str): Device for computations
        background_color (tuple|list|None): Optional (R, G, B) in [0,1]

    Returns:
        tuple: (renderer, lights)
    """

    # Create lighting based on preserve_vertex_colors flag
    if neutral_light:
        lights = create_neutral_lights(device=device)
    else:
        # Place the light relative to the camera pose. By default, offset is
        # along the camera's forward axis similar to the origin-based setup.
        lights = create_camera_relative_lights(cameras, device=device, offset_cam=(0.0, 0.0, -3.0))

    if renderer_type == RendererType.SOFT_PHONG:
        if background_color is None:
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        else:
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=background_color)

        raster_settings = create_raster_settings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(
                device=device, cameras=cameras, lights=lights, blend_params=blend_params
            ),
        )
    elif renderer_type == RendererType.SOFT_SILHOUETTE:
        if background_color is None:
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        else:
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=background_color)

        raster_settings = create_raster_settings(
            image_size=image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            faces_per_pixel=100,
            # bin_size=None,
            # max_faces_per_bin=50000,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )
    elif renderer_type == RendererType.HARD_DEPTH:
        if background_color is None:
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        else:
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=background_color)

        raster_settings = create_raster_settings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardDepthShader(device=device, cameras=cameras, blend_params=blend_params),
        )
    elif renderer_type == RendererType.HARD_PHONG:
        if background_color is None:
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        else:
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=background_color)

        raster_settings = create_raster_settings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(
                device=device, cameras=cameras, lights=lights, blend_params=blend_params
            ),
        )
    else:
        raise ValueError(f"Unsupported renderer type: {renderer_type}")

    return renderer


def render_frame(
    meshes,
    cameras,
    renderer,
    require_grad=False,
):
    # Render
    if not require_grad:
        with torch.no_grad():
            images = renderer(meshes, cameras=cameras)
    else:
        images = renderer(meshes, cameras=cameras)

    image = images[0, ..., :3]
    alpha = images[0, ..., 3]

    return image, alpha


def get_visible_verts(verts, faces, camera, image_size=None, rasterizer=None, mesh=None, zbuf=None):
    """
    Get visible vertices and their corresponding pixel positions in the projected image space.

    Args:
        verts (torch.Tensor): Mesh vertices of shape (V, 3)
        faces (torch.Tensor): Mesh faces of shape (F, 3)
        camera: PyTorch3D camera object
        image_size (tuple|None): Optional (H, W) to avoid querying camera each call
        rasterizer (MeshRasterizer|None): Optional prebuilt rasterizer to reuse
        mesh (Meshes|None): Optional prebuilt Meshes to reuse
        zbuf (torch.Tensor|None): Optional precomputed z-buffer of shape (H, W); skips rasterization

    Returns:
        tuple: (visible_vertices, pixel_positions)
            - visible_vertices (torch.Tensor): Visible vertices (N_visible, 3)
            - pixel_positions (torch.Tensor): 2D pixel positions of visible vertices (N_visible, 2)
    """
    device = verts.device

    with torch.inference_mode():
        # Transform vertices to different coordinate spaces
        verts_camera = (
            camera.get_world_to_view_transform().transform_points(verts.unsqueeze(0)).squeeze(0)
        )
        verts_screen = camera.transform_points_screen(verts.unsqueeze(0)).squeeze(0)

        # Determine image size (H, W)
        if image_size is None:
            cam_image_size = camera.get_image_size()
            H, W = cam_image_size[0]
            H = int(H.item())
            W = int(W.item())
        else:
            H, W = int(image_size[0]), int(image_size[1])

        # Extract pixel coordinates and depth
        pixel_coords = verts_screen[:, :2]  # (V, 2) - (x, y) pixel coordinates
        depths_camera = verts_camera[
            :, 2
        ]  # (V,) - depth values in camera space (for comparison with zbuf)

        # Check which vertices are within image bounds
        x_valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W)
        y_valid = (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H)
        in_bounds = x_valid & y_valid

        # Check which vertices are in front of camera (positive depth)
        in_front = depths_camera > 0

        # Combine basic visibility checks
        potentially_visible = in_bounds & in_front

        if not potentially_visible.any():
            # No vertices are potentially visible
            return torch.tensor([], dtype=torch.float32, device=device).reshape(0, 3), torch.tensor(
                [], dtype=torch.float32, device=device
            ).reshape(0, 2)

        # For vertices that pass basic checks, we need to check for occlusion
        # Optionally reuse provided z-buffer / rasterizer / mesh
        if zbuf is None:
            # Create mesh object if not provided
            if mesh is None:
                mesh = Meshes(verts=[verts], faces=[faces])

            # Create rasterization settings (optimized)
            raster_settings = RasterizationSettings(
                image_size=(int(H), int(W)),
                blur_radius=0.0,
                faces_per_pixel=1,  # Only keep closest face
                bin_size=0,  # disable binning because we have simplified the mesh
                max_faces_per_bin=50000,
                cull_backfaces=True,  # Reduce raster load
            )

            # Create or reuse rasterizer
            if rasterizer is None:
                rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

            # Rasterize the mesh to get fragment information
            fragments = rasterizer(mesh)

            # Get the depth buffer - shape (1, H, W, 1)
            zbuf = fragments.zbuf.squeeze()  # (H, W)
        else:
            # Ensure expected shape (H, W)
            if zbuf.dim() == 3:
                zbuf = zbuf.squeeze()

        # For each potentially visible vertex, check if it's occluded
        visible_mask = torch.zeros_like(potentially_visible, dtype=torch.bool)

        # Indices of potentially visible vertices
        potentially_visible_indices = potentially_visible.nonzero(as_tuple=True)[0]

        if len(potentially_visible_indices) > 0:
            # Vectorized processing of all potentially visible vertices
            visible_pixel_coords = pixel_coords[potentially_visible_indices]  # (N, 2)
            visible_depths_camera = depths_camera[potentially_visible_indices]  # (N,)

            # Round and clamp pixel coordinates to image bounds
            x_pixels = torch.clamp(visible_pixel_coords[:, 0].round().long(), 0, W - 1)  # (N,)
            y_pixels = torch.clamp(visible_pixel_coords[:, 1].round().long(), 0, H - 1)  # (N,)

            # Get surface depths at all pixel locations in one operation
            surface_depths = zbuf[
                y_pixels, x_pixels
            ]  # (N,) - depth buffer values at vertex pixel locations

            # Check visibility conditions vectorized
            depth_tolerance = 1e-2
            surface_valid = surface_depths > 0  # (N,) - surface exists at this pixel
            depth_close = (
                torch.abs(visible_depths_camera - surface_depths) <= depth_tolerance
            )  # (N,) - vertex is close to surface

            # Combine conditions to determine visibility
            vertex_visible = surface_valid & depth_close  # (N,)

            # Update the visibility mask for the potentially visible vertices
            visible_mask[potentially_visible_indices] = vertex_visible

        # Get indices of visible vertices
        visible_indices = visible_mask.nonzero(as_tuple=True)[0]

    # Get visible vertices and their corresponding pixel positions
    visible_verts = verts[visible_indices]
    visible_pixel_positions = pixel_coords[visible_indices].long()

    return visible_verts, visible_pixel_positions
