# src/ridges.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, peak_local_max
from configs.config import HESSIAN_SCALE_PX, RIDGE_SCALE_STEPS, RIDGE_SCALE_FACTOR, EPS
#TODO: scalling wit recompute hessian for each scale, change sigma at each scale
@dataclass
class RidgeMap:
    I: np.ndarray  # input image (row, col)
    hessian_scale_px: float = HESSIAN_SCALE_PX
    scale_steps: int = RIDGE_SCALE_STEPS
    scale_factor: float = RIDGE_SCALE_FACTOR

    # ---- computed fields (filled in __post_init__) ----
    vesselness: np.ndarray | None = field(init=False, default=None)
    thetas: np.ndarray | None = field(init=False, default=None)
    anisotropies: np.ndarray | None = field(init=False, default=None)
    best_hessian_scale: np.ndarray | None = field(init=False, default=None)
    lambdas_perp: np.ndarray | None = field(init=False, default=None)
    lambdas_par: np.ndarray | None = field(init=False, default=None)

    Hrr: np.ndarray | None = field(init=False, default=None)
    Hrc: np.ndarray | None = field(init=False, default=None)
    Hcc: np.ndarray | None = field(init=False, default=None)


    def __post_init__(self) -> None:
        # normalize input dtype once
        #self.I = np.asarray(self.I, dtype=np.float32)

        # force computation at construction
        self._compute_ridge_maps()
 # ------------------------------------------------------------------
    # internal computation
    # ------------------------------------------------------------------

    def _compute_ridge_maps(self) -> None:
        """
        Compute Hessian-based ridge maps at fixed scale configuration.
        Mutates self, returns nothing (Style 2).
        """

        # --- Hessian ---
        self.Hrr, self.Hrc, self.Hcc = hessian_matrix(
            self.I,
            sigma=self.hessian_scale_px,
            order="rc",
            use_gaussian_derivatives=True,
        )

        # --- orientation ---
        self.thetas = self._theta_from_hessian_components()

         # --- eigen values ---
        self.lambdas_perp, self.lambdas_par = hessian_eigenvalues_2d(
            self.Hrr,
            self.Hrc,
            self.Hcc,
        )


        # --- anisotropy ---
        self.anisotropies = np.abs(self.lambdas_perp) / (np.abs(self.lambdas_par) + EPS)

        # --- simple ridge strength (placeholder vesselness) ---
        # TODO: will replace this with full Frangi-style vesselness later
        vesselness = np.abs(self.lambdas_perp)

        if not np.isfinite(vesselness).all():
            raise ValueError("RidgeMap produced non-finite vesselness values")

        vesselness = np.maximum(vesselness, EPS).astype(np.float32)
        self.vesselness = vesselness
        # TODO: add best_hessian_scale for multiscale later
        # --- scale bookkeeping (single-scale baseline) ---
        self.best_hessian_scale = np.full_like(self.vesselness, self.hessian_scale_px, dtype=np.float32)

    # ------------------------------------------------------------------

    def _theta_from_hessian_components(self) -> np.ndarray:
        """
        Ridge tangent orientation in [0, pi).
        """
        phi = 0.5 * np.arctan2(
            2.0 * self.Hrc,
            (self.Hrr - self.Hcc) + EPS,
        )
        theta = phi + (np.pi / 2.0)
        theta = np.mod(theta, np.pi)
        return theta.astype(np.float32)

def hessian_eigenvalues_2d(
        Hrr: np.ndarray,
        Hrc: np.ndarray,
        Hcc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form eigenvalues of a 2x2 symmetric Hessian.

    Returns:
        lam_perp : eigenvalue with larger |magnitude|
        lam_par  : eigenvalue with smaller |magnitude|
    """

    # trace and determinant
    trace = Hrr + Hcc
    det = Hrr * Hcc - Hrc * Hrc

    # discriminant (guarded)
    delta = np.sqrt(np.maximum(trace * trace - 4.0 * det, 0.0))

    # eigenvalues
    lambda1 = 0.5 * (trace + delta)
    lambda2 = 0.5 * (trace - delta)

    # order by absolute value
    abs1 = np.abs(lambda1)
    abs2 = np.abs(lambda2)

    lam_perp = np.where(abs1 >= abs2, lambda1, lambda2)
    lam_par = np.where(abs1 < abs2, lambda1, lambda2)

    return lam_perp, lam_par
