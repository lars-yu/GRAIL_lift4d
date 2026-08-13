import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras


def cam_pose_blender_to_opencv(cam_R, cam_t):
    # Convert from Blender to OpenCV coordinate system
    # Blender: Y-up, Z-forward
    # OpenCV: Y-down, Z-forward
    cam_blender_to_opencv = torch.tensor(
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=cam_R.device, dtype=cam_R.dtype
    )
    cam_R = cam_R @ cam_blender_to_opencv

    return cam_R, cam_t.clone()


def cam_pose_opencv_to_pytorch3d(cam_R, cam_t):
    # Convert from OpenCV to PyTorch3D coordinate system
    # OpenCV: Y-down, Z-forward
    # PyTorch3D: Y-up, Z-back
    cam_opencv_to_pytorch3d = torch.tensor(
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], device=cam_R.device, dtype=cam_R.dtype
    )
    cam_R = cam_R @ cam_opencv_to_pytorch3d

    return cam_R, cam_t.clone()


def camera_to_world_matrix(cam_R: torch.Tensor, cam_t: torch.Tensor):
    c2w = torch.eye(4, device=cam_R.device, dtype=cam_R.dtype)
    c2w[:3, :3] = cam_R
    c2w[:3, 3] = cam_t.reshape((3,))
    return c2w


def world_to_camera_matrix(cam_R: torch.Tensor, cam_t: torch.Tensor):
    c2w = camera_to_world_matrix(cam_R, cam_t)
    w2c = torch.linalg.inv(c2w)
    return w2c


def transform_camera_to_world(verts: torch.Tensor, cam_R: torch.Tensor, cam_t: torch.Tensor):
    c2w = camera_to_world_matrix(cam_R, cam_t)
    verts = verts @ c2w[:3, :3].T + c2w[:3, 3]
    return verts


def transform_world_to_camera(verts: torch.Tensor, cam_R: torch.Tensor, cam_t: torch.Tensor):
    w2c = world_to_camera_matrix(cam_R, cam_t)
    verts = verts @ w2c[:3, :3].T + w2c[:3, 3]
    return verts


def transform_pose_c2w(pose_c: torch.Tensor, cam_R: torch.Tensor, cam_t: torch.Tensor):
    c2w = camera_to_world_matrix(cam_R, cam_t)
    pose_w = c2w @ pose_c
    return pose_w


def project_world_to_view(verts, camera):
    verts_camera = (
        camera.get_world_to_view_transform().transform_points(verts.unsqueeze(0)).squeeze(0)
    )

    return verts_camera


def project_world_to_screen(verts, camera):
    verts_screen = camera.transform_points_screen(verts.unsqueeze(0)).squeeze(0)
    return verts_screen


def unproject_depth_map_to_world(depth_map, cameras):
    """
    Unproject depth map to world space coordinates.

    Args:
        depth_map (torch.Tensor): Pixel coordinates with depth of shape (N, 3)
                                   where each row is [x_pixel, y_pixel, depth]
        cameras: PyTorch3D camera object

    Returns:
        torch.Tensor: World space coordinates of shape (N, 3)
    """
    device = depth_map.device
    N = depth_map.shape[0]

    if N == 0:
        return torch.tensor([], dtype=torch.float32, device=device).reshape(0, 3)

    # Extract pixel coordinates and depth
    xy_pixels = depth_map[:, :2]  # (N, 2) - pixel coordinates
    depths = depth_map[:, 2]  # (N,) - depth values

    # Combine pixel coordinates with depth to create 3D points in screen space
    # For PyTorch3D, screen space has format [x_pixel, y_pixel, depth]
    screen_points = torch.cat([xy_pixels, depths.unsqueeze(1)], dim=1)  # (N, 3)

    # Add batch dimension for camera processing
    screen_points_batched = screen_points.unsqueeze(0)  # (1, N, 3)

    # Step 1: Convert from screen space to camera space
    # We need to manually convert from screen to camera coordinates
    # Get camera intrinsics
    focal_length = cameras.focal_length[0]  # Assuming single camera
    principal_point = cameras.principal_point[0]  # Assuming single camera
    image_size = cameras.get_image_size()

    H, W = image_size[0]  # (height, width)
    H, W = int(H.item()), int(W.item())

    # Convert to PyTorch3D's coordinate system
    if isinstance(focal_length, torch.Tensor):
        if focal_length.dim() == 0:
            fx = fy = focal_length.item()
        else:
            fx, fy = focal_length[0].item(), (
                focal_length[1].item() if len(focal_length) > 1 else focal_length[0].item()
            )
    else:
        fx = fy = focal_length

    if isinstance(principal_point, torch.Tensor):
        px, py = principal_point[0].item(), principal_point[1].item()
    else:
        px, py = principal_point[0], principal_point[1]

    # Convert screen coordinates to camera coordinates
    x_pixel, y_pixel = xy_pixels[:, 0], xy_pixels[:, 1]

    # In PyTorch3D screen space, the depth is already in camera space units
    z_camera = depths

    # Convert pixel coordinates to PyTorch3D view-space coordinates.
    # PyTorch3D view space: +X left, +Y up, +Z toward camera.
    # Screen/image space: +X right, +Y down.
    # Negate both axes to match PyTorch3D convention.
    x_camera = -(x_pixel - px) * z_camera / fx
    y_camera = -(y_pixel - py) * z_camera / fy

    # Create camera space points
    camera_points = torch.stack([x_camera, y_camera, z_camera], dim=1)  # (N, 3)
    camera_points_batched = camera_points.unsqueeze(0)  # (1, N, 3)

    # Step 2: Convert from camera space to world space
    # Get the camera-to-world transformation
    camera_to_world_transform = cameras.get_world_to_view_transform().inverse()

    # Apply the transformation
    world_points_batched = camera_to_world_transform.transform_points(camera_points_batched)
    world_points = world_points_batched.squeeze(0)  # (N, 3)

    return world_points


def get_camera(cam_R, cam_T, focal_length, image_size, device="cuda", in_ndc=False):
    """
    Create a PyTorch3D camera with specified parameters

    Args:
        R (torch.Tensor): Rotation matrix (3x3)
        T (torch.Tensor): Translation vector (3,)
        focal_length (float): Camera focal length
        image_size (tuple): Image size (height, width)
        device (str): Device for computations
        in_ndc (bool): Whether to use NDC coordinates

    Returns:
        PerspectiveCameras: PyTorch3D camera object
    """

    # Ensure proper dimensions
    w2c = world_to_camera_matrix(cam_R, cam_T)
    w2c_R = w2c[:3, :3]
    w2c_T = w2c[:3, 3]

    if w2c_R.dim() == 2:
        w2c_R = w2c_R.unsqueeze(0)  # (1, 3, 3)
    if w2c_T.dim() == 1:
        w2c_T = w2c_T.unsqueeze(0)  # (1, 3)

    principal_point = ((image_size[1] / 2, image_size[0] / 2),)
    image_size = (image_size,)

    # Note: X_cam = X_world R + T so we need to transpose R to match the convention
    camera = PerspectiveCameras(
        R=w2c_R.transpose(1, 2),
        T=w2c_T,
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=image_size,
        device=device,
        in_ndc=in_ndc,
    )

    return camera
