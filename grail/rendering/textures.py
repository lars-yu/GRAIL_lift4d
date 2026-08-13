import torch
import torch.nn.functional as F
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes


def convert_textures_uv_to_vertex(mesh, device="cuda"):
    """
    Convert TexturesUV to TexturesVertex by sampling UV texture at vertex locations

    Args:
        mesh: PyTorch3D mesh with TexturesUV
        device: Device for computations

    Returns:
        TexturesVertex: Converted vertex textures
    """
    textures = mesh.textures

    # Get texture components
    texture_maps = textures.maps_padded()  # (N, H, W, 3) texture images
    verts_uvs = textures.verts_uvs_padded()  # (N, V, 2) UV coordinates for vertices

    if texture_maps is None or verts_uvs is None:
        print("Texture maps or verts_uvs is None, returning None")
        return None

    # Sample texture at vertex UV coordinates
    # texture_maps shape: (N, H, W, 3) where N is number of texture maps
    # verts_uvs shape: (N, V, 2) where N is batch size, V is number of vertices

    # Use the first batch and first texture map
    texture_map = texture_maps[0]  # (H, W, 3)
    vertex_uvs = verts_uvs[0]  # (V, 2)
    H, W = texture_map.shape[:2]

    # Convert UV coordinates to grid_sample format
    # UV coordinates are in [0, 1], grid_sample expects [-1, 1]
    # Also need to handle the coordinate system: UV (0,0) is top-left, but grid_sample (0,0) is center

    # Convert UV to grid coordinates
    grid_x = vertex_uvs[:, 0] * 2.0 - 1.0  # U: [0,1] -> [-1,1]
    grid_y = vertex_uvs[:, 1] * 2.0 - 1.0  # V: [0,1] -> [-1,1]

    # Stack to create grid: (V, 2)
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (V, 2)

    # Reshape for grid_sample: need (1, V, 1, 2) for batch_size=1, height=V, width=1
    grid = grid.unsqueeze(0).unsqueeze(2)  # (1, V, 1, 2)

    # Reshape texture for grid_sample: (1, 3, H, W)
    texture_for_sampling = texture_map.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # Sample texture
    sampled_colors = F.grid_sample(
        texture_for_sampling,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (1, 3, V, 1)

    # Reshape to vertex colors: (V, 3)
    vertex_colors = sampled_colors.squeeze(0).squeeze(-1).transpose(0, 1)  # (V, 3)

    # Create TexturesVertex
    vertex_textures = TexturesVertex(verts_features=vertex_colors.unsqueeze(0))

    return vertex_textures


def create_colored_meshes(verts, faces, color, device="cuda"):
    """
    Create colored meshes for human and object, with support for object textures
    """
    color = torch.as_tensor(color, dtype=torch.float32, device=device)
    vertex_colors = color.unsqueeze(0).repeat(verts.shape[0], 1)
    vertex_textures = TexturesVertex(verts_features=vertex_colors.unsqueeze(0))
    return Meshes(verts=[verts], faces=[faces], textures=vertex_textures)


def create_mesh_with_vertex_colors(verts, faces, vertex_colors, device="cuda"):
    """
    Create mesh with per-vertex colors.

    Args:
        verts: Vertex positions (N, 3) tensor
        faces: Face indices (F, 3) tensor
        vertex_colors: Per-vertex RGB colors (N, 3) tensor
        device: Device for computation

    Returns:
        Meshes object with per-vertex coloring
    """
    vertex_colors = torch.as_tensor(vertex_colors, dtype=torch.float32, device=device)
    vertex_textures = TexturesVertex(verts_features=vertex_colors.unsqueeze(0))
    return Meshes(verts=[verts], faces=[faces], textures=vertex_textures)
