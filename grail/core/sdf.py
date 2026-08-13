"""
Signed Distance Field (SDF) utilities for penetration loss.

This module provides functions to construct and query SDF fields from meshes,
enabling differentiable penetration loss during optimization.
"""

from typing import Optional, Tuple

import numpy as np
import torch


class MeshSDF:
    """
    Signed Distance Field representation of a mesh.
    Supports both grid-based (fast) and mesh-based (accurate) queries.
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        grid_resolution: int = 64,
        padding: float = 0.3,
        device: str = "cuda",
    ):
        """
        Initialize SDF from mesh.

        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (F, 3)
            grid_resolution: Resolution of SDF grid (higher = more accurate but slower, default 64)
            padding: Extra space around mesh as fraction of bounding box (default 0.3 = 30% padding on each side)
            device: Device for computation
        """
        self.vertices = vertices.to(device)
        self.faces = faces.to(device)
        self.device = device
        self.grid_resolution = grid_resolution

        print(f"Initializing MeshSDF with {len(vertices)} vertices, {len(faces)} faces")
        print(
            f"Grid resolution: {grid_resolution}x{grid_resolution}x{grid_resolution} = {grid_resolution**3} points"
        )

        # Compute bounding box with padding
        mesh_bbox_min = vertices.min(dim=0).values
        mesh_bbox_max = vertices.max(dim=0).values
        bbox_size = mesh_bbox_max - mesh_bbox_min
        padding_vec = bbox_size * padding
        self.bbox_min = mesh_bbox_min - padding_vec
        self.bbox_max = mesh_bbox_max + padding_vec
        self.bbox_size = self.bbox_max - self.bbox_min

        # Build SDF grid
        self.sdf_grid = self._build_sdf_grid()

    def _build_sdf_grid(self) -> torch.Tensor:
        """
        Build a 3D grid of SDF values.
        Uses mesh_to_sdf for accurate signed distance computation.

        Returns:
            SDF grid (res, res, res) where negative = inside, positive = outside
        """
        print(f"Building SDF grid with resolution {self.grid_resolution}...")

        # Create grid coordinates
        res = self.grid_resolution
        x = torch.linspace(0, 1, res, device=self.device)
        y = torch.linspace(0, 1, res, device=self.device)
        z = torch.linspace(0, 1, res, device=self.device)

        # Create meshgrid
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (res, res, res, 3)

        # Convert normalized coordinates to world space
        grid_points_world = grid_points * self.bbox_size + self.bbox_min

        # Compute SDF for all grid points
        # Flatten for batch processing
        grid_points_flat = grid_points_world.reshape(-1, 3)  # (res^3, 3)

        # Compute unsigned distance to mesh in batches
        batch_size = 5000  # Reduced batch size to avoid memory issues
        sdf_values = []
        signs_list = []

        print(f"Processing {len(grid_points_flat)} grid points in batches of {batch_size}...")
        for i in range(0, len(grid_points_flat), batch_size):
            batch_points = grid_points_flat[i : i + batch_size]
            batch_sdf = self._compute_unsigned_distance_to_mesh(batch_points)
            batch_signs = self._compute_signs(batch_points)
            sdf_values.append(batch_sdf)
            signs_list.append(batch_signs)

            # Clear cache periodically
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

        sdf_values = torch.cat(sdf_values, dim=0)
        signs = torch.cat(signs_list, dim=0)
        sdf_values = sdf_values * signs

        # Reshape back to grid
        sdf_grid = sdf_values.reshape(res, res, res)

        print(f"SDF grid built. Value range: [{sdf_grid.min():.4f}, {sdf_grid.max():.4f}]")
        return sdf_grid

    def _compute_unsigned_distance_to_mesh(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute unsigned distance from points to mesh surface using exact
        point-to-triangle distances (Ericson's closest-point algorithm).

        Args:
            points: Query points (N, 3)

        Returns:
            Distances (N,)
        """
        v0 = self.vertices[self.faces[:, 0]]  # (F, 3)
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        F = v0.shape[0]
        N = points.shape[0]

        sub_batch_size = max(1, min(500, 40_000_000 // max(F, 1)))

        all_min_dists = []
        for i in range(0, N, sub_batch_size):
            bp = points[i : i + sub_batch_size]  # (B, 3)

            p = bp.unsqueeze(1)  # (B, 1, 3)
            a = v0.unsqueeze(0)  # (1, F, 3)
            b = v1.unsqueeze(0)
            c = v2.unsqueeze(0)

            ab = b - a  # (1, F, 3)
            ac = c - a
            ap = p - a  # (B, F, 3)
            bp_ = p - b
            cp = p - c

            d1 = (ab * ap).sum(-1)  # (B, F)
            d2 = (ac * ap).sum(-1)
            d3 = (ab * bp_).sum(-1)
            d4 = (ac * bp_).sum(-1)
            d5 = (ab * cp).sum(-1)
            d6 = (ac * cp).sum(-1)

            vc = d1 * d4 - d3 * d2
            vb = d5 * d2 - d1 * d6
            va = d3 * d6 - d5 * d4

            # Region 7 (interior of triangle) as default
            denom = 1.0 / (va + vb + vc + 1e-12)
            v_param = vb * denom
            w_param = vc * denom
            closest = a + v_param.unsqueeze(-1) * ab + w_param.unsqueeze(-1) * ac

            # Override with edge / vertex regions (low-to-high priority)
            # Region 6: edge BC
            w6 = ((d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-12)).clamp(0, 1)
            m6 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
            closest = torch.where(m6.unsqueeze(-1), b + w6.unsqueeze(-1) * (c - b), closest)

            # Region 5: edge AC
            w5 = (d2 / (d2 - d6 + 1e-12)).clamp(0, 1)
            m5 = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
            closest = torch.where(m5.unsqueeze(-1), a + w5.unsqueeze(-1) * ac, closest)

            # Region 4: vertex C
            m4 = (d6 >= 0) & (d5 <= d6)
            closest = torch.where(m4.unsqueeze(-1), c.expand_as(closest), closest)

            # Region 3: edge AB
            v3 = (d1 / (d1 - d3 + 1e-12)).clamp(0, 1)
            m3 = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
            closest = torch.where(m3.unsqueeze(-1), a + v3.unsqueeze(-1) * ab, closest)

            # Region 2: vertex B
            m2 = (d3 >= 0) & (d4 <= d3)
            closest = torch.where(m2.unsqueeze(-1), b.expand_as(closest), closest)

            # Region 1: vertex A (highest priority)
            m1 = (d1 <= 0) & (d2 <= 0)
            closest = torch.where(m1.unsqueeze(-1), a.expand_as(closest), closest)

            sq_dist = ((p - closest) ** 2).sum(-1)  # (B, F)
            min_dist = sq_dist.min(dim=1).values.sqrt()  # (B,)
            all_min_dists.append(min_dist)

        return torch.cat(all_min_dists, dim=0)

    def _compute_signs(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute inside/outside signs using the generalized winding number
        (Van Oosterom & Strackee solid-angle formula).
        Robust for non-convex and complex meshes.

        Args:
            points: Query points (N, 3)

        Returns:
            Signs (N,) - +1 for outside, -1 for inside
        """
        v0 = self.vertices[self.faces[:, 0]]  # (F, 3)
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        F = v0.shape[0]
        N = points.shape[0]

        sub_batch_size = max(1, min(500, 40_000_000 // max(F, 1)))

        all_signs = []
        for i in range(0, N, sub_batch_size):
            bp = points[i : i + sub_batch_size]  # (B, 3)

            # Vectors from query point to each triangle vertex
            a = v0.unsqueeze(0) - bp.unsqueeze(1)  # (B, F, 3)
            b = v1.unsqueeze(0) - bp.unsqueeze(1)
            c = v2.unsqueeze(0) - bp.unsqueeze(1)

            la = a.norm(dim=-1)  # (B, F)
            lb = b.norm(dim=-1)
            lc = c.norm(dim=-1)

            # Solid angle: Omega = 2 * atan2( a . (b x c), denom )
            cross_bc = torch.cross(b, c, dim=-1)  # (B, F, 3)
            triple = (a * cross_bc).sum(-1)  # (B, F)

            denom = (
                la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (a * c).sum(-1) * lb
            )

            omega = 2.0 * torch.atan2(triple, denom)  # (B, F)
            winding = omega.sum(dim=1) / (4.0 * np.pi)  # (B,)

            signs = torch.where(
                winding.abs() > 0.5,
                torch.tensor(-1.0, device=points.device),
                torch.tensor(1.0, device=points.device),
            )
            all_signs.append(signs)

        return torch.cat(all_signs, dim=0)

    def _compute_vertex_normals(self) -> torch.Tensor:
        """
        Compute per-vertex normals from face normals.

        Returns:
            Vertex normals (V, 3)
        """
        # Get vertices for each face
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        # Compute face normals
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-8)

        # Accumulate face normals to vertices
        vertex_normals = torch.zeros_like(self.vertices)
        vertex_normals.index_add_(0, self.faces[:, 0], face_normals)
        vertex_normals.index_add_(0, self.faces[:, 1], face_normals)
        vertex_normals.index_add_(0, self.faces[:, 2], face_normals)

        # Normalize
        vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-8)

        return vertex_normals

    def query(self, points: torch.Tensor, method: str = "grid") -> torch.Tensor:
        """
        Query SDF values at given points.

        Args:
            points: Query points (N, 3) or (B, N, 3)
            method: "grid" (fast, trilinear interpolation) or "mesh" (accurate, direct computation)

        Returns:
            SDF values (N,) or (B, N) - negative inside, positive outside
        """
        if method == "grid":
            return self._query_grid(points)
        elif method == "mesh":
            return self._query_mesh(points)
        else:
            raise ValueError(f"Unknown query method: {method}")

    def _query_grid(self, points: torch.Tensor) -> torch.Tensor:
        """
        Query SDF using trilinear interpolation on pre-computed grid.
        Fast and differentiable.

        Args:
            points: Query points (N, 3) or (B, N, 3)

        Returns:
            SDF values (N,) or (B, N)
        """
        original_shape = points.shape
        if points.dim() == 2:
            points = points.unsqueeze(0)  # Add batch dimension

        batch_size, num_points, _ = points.shape

        # Normalize points to [0, 1]
        normalized_points = (points - self.bbox_min) / self.bbox_size
        normalized_points = torch.clamp(normalized_points, 0, 1)

        # Correct reshaping for grid_sample
        # Input: (B, C, D, H, W) - our sdf_grid is (res, res, res), need to add batch and channel
        # Grid: (B, D_out, H_out, W_out, 3)
        sdf_grid_batched = (
            self.sdf_grid.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        )  # (B, 1, res, res, res)

        # Reshape query points for grid_sample
        # We want output shape (B, num_points), so grid should be (B, num_points, 1, 1, 3)
        grid_coords_reshaped = normalized_points * 2 - 1  # (B, N, 3) -> range [-1, 1]
        # grid_sample convention: coord[...,0]->W, coord[...,1]->H, coord[...,2]->D
        # Our sdf_grid after unsqueeze is (B,C,D=x,H=y,W=z), so pass (z,y,x)
        grid_coords_reshaped = grid_coords_reshaped[..., [2, 1, 0]]
        grid_coords_reshaped = grid_coords_reshaped.view(batch_size, num_points, 1, 1, 3)

        # Sample from grid
        sampled = torch.nn.functional.grid_sample(
            sdf_grid_batched,
            grid_coords_reshaped,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (B, 1, num_points, 1, 1)

        # Reshape output
        sdf_values = sampled.squeeze(1).squeeze(-1).squeeze(-1)  # (B, num_points)

        # Return original shape
        if len(original_shape) == 2:
            sdf_values = sdf_values.squeeze(0)  # Remove batch dimension

        return sdf_values

    def _query_mesh(self, points: torch.Tensor) -> torch.Tensor:
        """
        Query SDF by direct computation from mesh.
        More accurate but slower than grid query.

        Args:
            points: Query points (N, 3) or (B, N, 3)

        Returns:
            SDF values (N,) or (B, N)
        """
        original_shape = points.shape
        if points.dim() == 2:
            points = points.unsqueeze(0)

        # Compute unsigned distance
        unsigned_dist = self._compute_unsigned_distance_to_mesh(points.view(-1, 3))

        # Compute signs
        signs = self._compute_signs(points.view(-1, 3))

        # Signed distance
        sdf_values = unsigned_dist * signs
        sdf_values = sdf_values.view(original_shape[:-1])

        if len(original_shape) == 2:
            sdf_values = sdf_values.squeeze(0)

        return sdf_values


# Utility function for quick usage
def create_sdf_from_mesh(
    vertices: torch.Tensor, faces: torch.Tensor, grid_resolution: int = 128, device: str = "cuda"
) -> MeshSDF:
    """
    Convenience function to create SDF from mesh.

    Args:
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (F, 3)
        grid_resolution: Grid resolution (higher = more accurate, default 64 for memory efficiency)
        device: Device for computation

    Returns:
        MeshSDF object
    """
    return MeshSDF(vertices, faces, grid_resolution, device=device)
