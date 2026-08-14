"""3D-to-2D projection math and matrix utilities for Module 9."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_look_at_matrix(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Compute 4x4 View Matrix (World to Camera frame)."""
    eye = np.asarray(eye, dtype=np.float32).reshape(3)
    target = np.asarray(target, dtype=np.float32).reshape(3)
    up = np.asarray(up, dtype=np.float32).reshape(3)

    forward = target - eye
    norm_f = np.linalg.norm(forward)
    if norm_f < 1e-8:
        forward = np.array([0, 1, 0], dtype=np.float32)
    else:
        forward = forward / norm_f

    right = np.cross(forward, up)
    norm_r = np.linalg.norm(right)
    if norm_r < 1e-8:
        right = np.array([1, 0, 0], dtype=np.float32)
    else:
        right = right / norm_r

    true_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward  # Camera looks along -Z in standard camera space
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(true_up, eye)
    view[2, 3] = np.dot(forward, eye)

    return view


def compute_perspective_matrix(
    fov_degrees: float,
    aspect_ratio: float,
    near: float,
    far: float,
) -> np.ndarray:
    """Compute 4x4 Perspective Projection Matrix."""
    fov_rad = np.radians(fov_degrees)
    tan_half_fov = np.tan(fov_rad / 2.0)

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 1.0 / (aspect_ratio * tan_half_fov)
    proj[1, 1] = 1.0 / tan_half_fov
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2.0 * far * near) / (far - near)
    proj[3, 2] = -1.0

    return proj


def compute_orthographic_matrix(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> np.ndarray:
    """Compute 4x4 Orthographic Projection Matrix."""
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 2.0 / (right - left)
    proj[1, 1] = 2.0 / (top - bottom)
    proj[2, 2] = -2.0 / (far - near)
    proj[0, 3] = -(right + left) / (right - left)
    proj[1, 3] = -(top + bottom) / (top - bottom)
    proj[2, 3] = -(far + near) / (far - near)
    proj[3, 3] = 1.0

    return proj


def project_points(
    points_3d: np.ndarray,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project Nx3 3D world points to 2D pixel coordinates and depths.

    Returns
    -------
    pixel_coords : ndarray[M, 2]
        (u, v) pixel coordinates in [0, width-1] x [0, height-1].
    depths : ndarray[M]
        Camera-space depths (Z).
    valid_mask : ndarray[N] (bool)
        Mask of points that lie inside the view frustum (near < depth < far).
    """
    if len(points_3d) == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool),
        )

    # 1. Transform to Camera space
    pts_homo = np.hstack([points_3d, np.ones((len(points_3d), 1), dtype=np.float32)])
    cam_pts = (view_matrix @ pts_homo.T).T
    depths = -cam_pts[:, 2]  # positive in front of camera

    # 2. Transform to Clip space
    clip_pts = (proj_matrix @ cam_pts.T).T
    w = clip_pts[:, 3:4]

    # Valid mask: in front of near plane and not behind camera
    valid = (w[:, 0] > 1e-4) & (depths > 0)

    # 3. Normalized Device Coordinates (NDC) [-1, 1]
    ndc_x = np.zeros(len(points_3d), dtype=np.float32)
    ndc_y = np.zeros(len(points_3d), dtype=np.float32)
    ndc_x[valid] = clip_pts[valid, 0] / w[valid, 0]
    ndc_y[valid] = clip_pts[valid, 1] / w[valid, 0]

    # In-frustum mask
    in_frustum = valid & (ndc_x >= -1.0) & (ndc_x <= 1.0) & (ndc_y >= -1.0) & (ndc_y <= 1.0)

    # 4. Pixel Coordinates (u, v)
    u = ((ndc_x + 1.0) * 0.5 * (image_width - 1)).astype(np.int32)
    v = ((1.0 - ndc_y) * 0.5 * (image_height - 1)).astype(np.int32)  # Y inverted in image space

    u = np.clip(u, 0, image_width - 1)
    v = np.clip(v, 0, image_height - 1)

    pixel_coords = np.column_stack([u, v])
    return pixel_coords, depths, in_frustum
