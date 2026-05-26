"""Geometry helpers: quaternions, yaw extraction, global -> ego transforms.

Conventions
-----------
* Quaternions are stored as ``[w, x, y, z]`` (NuScenes / scipy ``scalar_first``).
* World/ego/lidar frames are right-handed, with ``+z`` up.
* "Yaw" means rotation around ``+z`` (counter-clockwise when viewed from above).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def quaternion_to_rotmat(q: Sequence[float]) -> np.ndarray:
    """Convert a quaternion ``[w, x, y, z]`` to a ``3x3`` rotation matrix.

    The result rotates a vector expressed in the body frame into the parent
    frame: ``v_parent = R @ v_body``.
    """
    w, x, y, z = q
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
            [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
            [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_to_yaw(q: Sequence[float]) -> float:
    """Yaw (rotation around +z) of a quaternion in radians."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrap_angle(a: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def transform_global_to_ego(
    translation: Sequence[float],
    rotation_q: Sequence[float],
    ego_translation: Sequence[float],
    ego_rotation_q: Sequence[float],
) -> tuple[np.ndarray, float]:
    """Transform a 3D box pose from the global frame into the ego frame.

    Parameters
    ----------
    translation : len-3
        Box center in global frame, meters.
    rotation_q : len-4
        Box orientation as a ``[w, x, y, z]`` quaternion (body -> global).
    ego_translation : len-3
        Ego pose translation in global frame.
    ego_rotation_q : len-4
        Ego orientation as a ``[w, x, y, z]`` quaternion (ego -> global).

    Returns
    -------
    pos_ego : ``(3,)`` float64
        Box center expressed in the ego frame.
    yaw_ego : float
        Box yaw around the ego ``+z`` axis, wrapped to ``(-pi, pi]``.
    """
    R_ge = quaternion_to_rotmat(ego_rotation_q)  # ego -> global
    delta = np.asarray(translation, dtype=np.float64) - np.asarray(
        ego_translation, dtype=np.float64
    )
    pos_ego = R_ge.T @ delta  # global -> ego

    yaw_global = quaternion_to_yaw(rotation_q)
    yaw_ego_pose = quaternion_to_yaw(ego_rotation_q)
    yaw_ego = wrap_angle(yaw_global - yaw_ego_pose)
    return pos_ego, yaw_ego
