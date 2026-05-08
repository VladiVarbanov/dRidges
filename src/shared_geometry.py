from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from configs.config import (EPS,
                            PARTNER_OFFSET_PX)

def derive_sigmas_from_local_geometry(
    *,
    base_scale_px: float,
    eigen_anisotropy: Optional[float] = None,
    lambda_perp: Optional[float] = None,
    lambda_para: Optional[float] = None,
    theta: Optional[float] = None,
    flat_area_cap: float,   #TODO: check if ised in ridge seeding
    eps: float
) -> Tuple[float, float]:
    """
    Derive anisotropic ellipse scales from local geometry.

    Parameters
    ----------
    base_scale_px
        Absolute normal-direction scale in pixels, chosen by the caller.
        In ridge seeding, this would come from tile-size policy.
    eigen_anisotropy
        Preferred anisotropy input. If None, it is computed from lambdas.
    lambda_perp, lambda_para
        Optional Hessian eigenvalues. Used only if eigen_anisotropy is None.
    theta
        Accepted for interface continuity and future extensions.
        Not used here yet, because theta orients the ellipse rather than
        setting its axis lengths.
    flat_area_cap
        Upper bound on tangent/normal ratio to avoid exploding sigmas in
        flat or weakly directional areas.
    eps
        Small positive guard.

    Returns
    -------
    sigma_n_px, sigma_t_px
        Ellipse scales in the local normal/tangent frame.
    """
    sigma_n_px = max(float(base_scale_px), float(eps))

    if eigen_anisotropy is None:
        if lambda_perp is None or lambda_para is None:
            raise ValueError(
                "Provide either eigen_anisotropy or both lambda_perp and lambda_para."
            )
        eigen_anisotropy = abs(float(lambda_perp)) / (abs(float(lambda_para)) + float(eps))

    if not np.isfinite(eigen_anisotropy):
        raise ValueError(f"Non-finite eigen_anisotropy: {eigen_anisotropy}")

    ratio_t_over_n = float(np.clip(float(eigen_anisotropy), 1.0, float(flat_area_cap)))
    sigma_t_px = max(sigma_n_px * ratio_t_over_n, float(eps))

    return sigma_n_px, sigma_t_px


def _wrap_angle_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi
def local_frame_from_theta(theta: float) -> tuple[float, float, float, float]:
    """
    Return local ridge-frame unit vectors from tangent angle theta.

    Parameters
    ----------
    theta
        Ridge tangent angle in radians.

    Returns
    -------
    t_row, t_col, n_row, n_col
        Unit tangent and unit normal components in image coordinates.

    Notes
    -----
    Image coordinates:
        - row increases downward
        - col increases rightward

    Convention used here matches anisotropic_distance2:
        tangent = (sin(theta),  cos(theta))
        normal  = (cos(theta), -sin(theta))
    """
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))

    t_row = sin_theta
    t_col = cos_theta
    n_row = cos_theta
    n_col = -sin_theta

    return t_row, t_col, n_row, n_col


def project_point_into_local_frame(
    row_ref: float,
    col_ref: float,
    t_row: float,
    t_col: float,
    n_row: float,
    n_col: float,
    row_test: float,
    col_test: float,
) -> tuple[float, float]:
    """
    Project a test point into the local tangent-normal frame of a reference point.

    Parameters
    ----------
    row_ref, col_ref
        Reference point coordinates.
    t_row, t_col
        Tangent unit-vector components of the reference local frame.
    n_row, n_col
        Normal unit-vector components of the reference local frame.
    row_test, col_test
        Test point coordinates.

    Returns
    -------
    d_t, d_n
        Coordinates of the reference-to-test vector in the local frame:
        - d_t: tangential projection
        - d_n: normal projection
    """
    row_diff = row_test - row_ref
    col_diff = col_test - col_ref

    d_t = row_diff * t_row + col_diff * t_col
    d_n = row_diff * n_row + col_diff * n_col

    return float(d_t), float(d_n)


def candidate_side_from_normal_projection(
    d_n: float,
    *,
    eps: float = 1e-6,
) -> int:
    """
    Classify which side of the reference normal a candidate lies on.

    Parameters
    ----------
    d_n
        Normal projection of the reference-to-candidate vector.
    eps
        Small tolerance for treating near-zero projection as ambiguous.

    Returns
    -------
    side
        +1 if candidate lies on the +normal side
        -1 if candidate lies on the -normal side
         0 if candidate is too close to the tangent line to decide
    """
    if d_n > eps:
        return 1
    if d_n < -eps:
        return -1
    return 0

def partner_gate_center(
    row_ref: float,
    col_ref: float,
    n_row: float,
    n_col: float,
    offset_px: float = PARTNER_OFFSET_PX ,
) -> tuple[float, float]:
    """
    Return the expected partner gate center for a reference point.

    Parameters
    ----------
    row_ref, col_ref
        Reference point coordinates.
    n_row, n_col
        Normal unit-vector components of the reference local frame.
    offset_px
        Signed offset from the reference point in pixels.

    Returns
    -------
    row_gate, col_gate
        Gate-center coordinates obtained by moving from the reference point
        along the local normal by offset_px.
    """
    row_gate = row_ref + offset_px * n_row
    col_gate = col_ref + offset_px * n_col
    return float(row_gate), float(col_gate)

def axial_angle_difference(theta_a: float, theta_b: float) -> float:
    """
    Return the smallest angular difference between two axial angles.

    Parameters
    ----------
    theta_a, theta_b
        Angles in radians representing axial directions, where theta and
        theta + pi are equivalent.

    Returns
    -------
    diff
        Smallest absolute angular difference in [0, pi/2].
    """
    diff = abs(_wrap_angle_pi(theta_a - theta_b))
    if diff > (np.pi / 2.0):
        diff = np.pi - diff
    return float(diff)


def axial_average_theta(theta_a: float, theta_b: float) -> float:
    """
    Return the axial mean of two ridge tangent angles.

    Parameters
    ----------
    theta_a, theta_b
        Ridge tangent angles in radians. Axial meaning theta and theta + pi
        represent the same direction.

    Returns
    -------
    theta_mean
        Axial-average angle wrapped to [0, pi).
    """
    sin2 = np.sin(2.0 * theta_a) + np.sin(2.0 * theta_b)
    cos2 = np.cos(2.0 * theta_a) + np.cos(2.0 * theta_b)

    theta_mean = 0.5 * np.arctan2(sin2, cos2)
    theta_mean = np.mod(theta_mean, np.pi)

    return float(theta_mean)


def anisotropic_distance2(
    row_ref: float, col_ref: float, theta_ref: float,
    row_test: float, col_test: float,
    sigma_n_px: float, sigma_t_px: float,
) -> float:
    """
    Compute d^2 in the ridge frame of seed0.
    theta0 is tangent direction.
    """
    row_diff = row_test - row_ref
    col_diff = col_test - col_ref

    cos_theta = float(np.cos(theta_ref))
    sin_theta = float(np.sin(theta_ref))

    # tangent unit vector (row, col) components:
    # Note: in image coordinates, row increases downward, col increases rightward.
    # This is fine as long as consistent.

    # unit tangent vector (|t| = 1)
    t_row, t_col = sin_theta, cos_theta
    # unit normal vector (|n| = 1,
    n_row, n_col = cos_theta, -sin_theta  # rotate tangent by -90 deg

    vec_t_px = row_diff * t_row + col_diff * t_col
    vec_n_px = row_diff * n_row + col_diff * n_col

    if sigma_n_px < -EPS or sigma_t_px < -EPS:
        raise ValueError("sigma must be non-negative (within tolerance)")
    # sigma_n and sigma_t are ellipse scale parameters in pixel units
    sigma_n_px = max(float(sigma_n_px), EPS)
    sigma_t_px = max(float(sigma_t_px), EPS)

    return (vec_n_px * vec_n_px) / (sigma_n_px * sigma_n_px) + (vec_t_px * vec_t_px) / (sigma_t_px * sigma_t_px)


