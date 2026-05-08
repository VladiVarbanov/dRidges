from __future__ import annotations

from typing import Literal
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple, List
import math

from .ridges import RidgeMap
from .shared_geometry import (anisotropic_distance2,
                             _wrap_angle_pi,
                             derive_sigmas_from_local_geometry,
                             project_point_into_local_frame,
                             local_frame_from_theta,
                             candidate_side_from_normal_projection,
                             axial_angle_difference,
                             axial_average_theta,
                             partner_gate_center,
                             )

import numpy as np

from configs.config import (HESSIAN_SCALE_PX,
                            RIDGE_SCALE_STEPS,
                            RIDGE_SCALE_FACTOR,
                            EPS, QUANTILE,
                            MAX_SEEDS,
                            SEEDS_PER_TILE,
                            TILE_DIVISOR,
                            TILE_MIN,
                            TILE_MAX,
                            BASE_SIGMA_N_DIVISOR,
                            BASE_SIGMA_N_PX_MIN,
                            BASE_SIGMA_N_PX_MAX,
                            FLAT_AREA_CAP,
                            SUPPRESSION_DISTANCE_THRESHOLD,
                            PARTNER_OFFSET_PX,
                            PARTNER_GATE_HALF_WIDTH_N_PX,
                            PARTNER_GATE_HALF_WIDTH_T_PX,
                            PARTNER_THETA_TOLERANCE,
                            PARTNER_REJECTION_DISTANCE_PX,
                            STRENGTH_WEIGHT,
                            GATE_CENTER_N_WEIGHT,
                            GATE_CENTER_T_WEIGHT,
                            THETA_WEIGHT,
                            )

#TODO: Ensure ridge spacing and strong quantile do not kill all the good seeds
@dataclass(frozen=True)
class Seed:
    seed_id: int
    row: int
    col: int

    # strength + ranking
    strength: float
    rank_percentile: float  # in [0, 1]

    # orientation + anisotropic covariance parameters (ridge frame)
    theta: float                     # radians, tangent direction
    sigma_n_px: float                   # pixels (normal)
    sigma_t_px: float                   # pixels (tangent)
    hessian_scale_px: float          # Hessian scale used to compute maps (metadata)

    # curvature proxies at seed time (Hessian eigenvalues in ridge frame)
    lambda_perp: float
    lambda_para: float
    eigen_anisotropy: float


@dataclass(frozen=True)
class Grid:
    grid_id: int
    offset_row: int
    offset_col: int
    tile_size: int

@dataclass
class GridPlacement:
    tile_row: int
    tile_col: int
    rank_percentile: float = -1.0


@dataclass
class SeedPlacement:
    seed_id: int
    row: int
    col: int
    placement_by_grid: dict[int, GridPlacement]
    survived_in_grids: list[int] = field(default_factory=list)

@dataclass
class CenterlineSeed:
    mid_seed_id: int
    mid_row: float
    mid_col: float
    mid_theta: float
    mid_strength: float
    mid_width_px: float

    source_anchor_seed_id: int
    source_partner_seed_id: int
    source_anchor_row: int
    source_anchor_col: int
    source_partner_row: int
    source_partner_col: int
    source_anchor_strength: float
    source_partner_strength: float

@dataclass(frozen=True)
class SuppressionRecord:
    # --- copied from Seed for self-contained CSV/debug ---
    seed_id: int
    row: int
    col: int
    strength: float
    rank_percentile: float
    theta: float
    sigma_n_px: float
    sigma_t_px: float
    hessian_scale_px: float
    lambda_perp: float
    lambda_para: float
    eigen_anisotropy: float

    # --- run/view context ---
    support_view: str  # "grid0", "grid1", "both", or "all"
    suppression_distance_threshold: float
    suppression_distance_threshold_sq: float

    # --- suppression outcome ---
    decision: Literal["kept", "suppressed"]
    kept_index: int | None
    num_compared_kept_seeds: int

    # --- nearest among compared kept seeds ---
    nearest_compared_seed_id: int | None
    nearest_compared_distance2: float | None
    nearest_compared_distance: float | None

    # --- actual blocker, only for suppressed seeds ---
    blocking_seed_id: int | None
    blocking_distance2: float | None
    blocking_distance: float | None
# Planting positions preparation
# -----------------------------

def select_strong_positions(
    strength_map: np.ndarray,
    *,
    q: float = QUANTILE,
) -> np.ndarray:
    """
    Boolean mask of strong candidates: strength_map >= quantile(strength_map, q).
    """
    strength_threshold = float(np.quantile(strength_map[np.isfinite(strength_map)], q))
    return (strength_map >= strength_threshold) & np.isfinite(strength_map)

# TODO: seed_per_tile maybe different from 1

def rank_strong_positions(strength_map: np.ndarray, seeds_mask: np.ndarray) -> np.ndarray:
    if strength_map.shape != seeds_mask.shape:
        raise ValueError(
            f"rank_strong_candidates expects strength_map and seeds_mask to have the same shape, "
            f"got strength_map.shape={strength_map.shape}, seeds_mask.shape={seeds_mask.shape}"
        )

    if seeds_mask.dtype != bool:
        raise TypeError(
            f"rank_strong_candidates expects seeds_mask to be boolean, "
            f"got dtype={seeds_mask.dtype}"
        )

    candidate_vals = strength_map[seeds_mask]
    candidate_inds = np.argsort(candidate_vals)

    ranks = np.empty(candidate_inds.size, dtype=np.float32)
    ranks[candidate_inds] = (
        np.arange(candidate_inds.size, dtype=np.float32)
        / max(candidate_inds.size - 1, 1)
    )
    return ranks

def map_ranked_positions(seeds_mask: np.ndarray, ranked_candidates: np.ndarray) -> np.ndarray:
    """gives the 2D image-shaped map output is used in many functions as  global_rank_map  param"""
    rank_map = np.full_like(seeds_mask, fill_value=-1, dtype=np.float32) # strength_map is shape template here
    rank_map[seeds_mask] = ranked_candidates.astype(np.float32)
    return rank_map


# -----------------------------
# Parcel: Partition image into tiles and grids
# -----------------------------

def choose_tile_size(src_img: np.ndarray,
                    *,
                    tile_divisor: int = TILE_DIVISOR ,       # tile_size ~= min(row,col)/tile_divisor
                    tile_min: int = TILE_MIN,
                    tile_max: int = TILE_MAX,
) -> int:
    rows, cols = src_img.shape
    initial_tile_size  = int(round(min(rows, cols) / float(tile_divisor)))
    tile_size = int(np.clip(initial_tile_size , tile_min, tile_max))
    return tile_size


def define_grids(tile_size: int) -> tuple[Grid, Grid]:
    half = tile_size // 2
    grid_a = Grid(grid_id=0, offset_row=0,    offset_col=0,    tile_size=tile_size)
    grid_b = Grid(grid_id=1, offset_row=half, offset_col=half, tile_size=tile_size)
    return grid_a, grid_b


def compute_tile_from_point_in_grid(row: int, col: int, grid: Grid) -> tuple[int, int]:
    tile_row = (row - grid.offset_row) // grid.tile_size
    tile_col = (col - grid.offset_col) // grid.tile_size
    return int(tile_row), int(tile_col)



# -----------------------------
# Seeding: Place the seeds in to grids
# -----------------------------
def create_seed_placements(
    seeds_mask: np.ndarray,
    grids: tuple[Grid, Grid],
) -> list[SeedPlacement]:
    """
    Create one physical seed per selected position, then place that same seed
    into each grid.

    seed_id is assigned once from the sequential order of seeds_mask.
    """
    rows, cols = np.nonzero(seeds_mask)

    placements: list[SeedPlacement] = []

    for seed_id, (row, col) in enumerate(zip(rows, cols)):
        row = int(row)
        col = int(col)

        placement = SeedPlacement(
            seed_id=seed_id,
            row=row,
            col=col,
            placement_by_grid ={}
        )

        for grid in grids:
            tile_row, tile_col = compute_tile_from_point_in_grid(row, col, grid)

            placement.placement_by_grid[grid.grid_id] = GridPlacement(
                tile_row=tile_row,
                tile_col=tile_col,
                rank_percentile=-1.0,
            )

        placements.append(placement)

    return placements

from collections import defaultdict
from typing import Iterable


def group_seed_placements_by_tile(
    placements: Iterable[SeedPlacement],
) -> dict[tuple[int, int, int], list[SeedPlacement]]:
    """
    Group placements by (grid_id, tile_row, tile_col).
    """
    placements_by_tile: dict[tuple[int, int, int], list[SeedPlacement]] = defaultdict(list)

    for placement in placements:
        for grid_id, grid_placement in placement.placement_by_grid.items():
            key = (
                int(grid_id),
                int(grid_placement.tile_row),
                int(grid_placement.tile_col),
            )
            placements_by_tile[key].append(placement)

    return dict(placements_by_tile)

def assign_local_rank_percentiles_in_place(
    placements_by_tile: dict[tuple[int, int, int], list[SeedPlacement]],
    strength_map: np.ndarray,
    global_rank_map: np.ndarray,
) -> None:
    """
    Compute tile-local rank_percentile for each placement in place.

    Ordering inside a tile:
      1. stronger strength first
      2. larger global rank first
      3. smaller row
      4. smaller col

    Rank is written into:
        placement.placement_by_grid[grid_id].rank_percentile

    Best in tile -> 1.0
    Worst in tile -> 0.0
    """
    for (grid_id, _, _), placements in placements_by_tile.items():
        if not placements:
            continue

        ordered = sorted(
            placements,
            key=lambda p: (     #TODO: Look closely at this function
                -float(strength_map[p.row, p.col]),
                -float(global_rank_map[p.row, p.col]),
                p.row,
                p.col,
            ),
        )

        n = len(ordered)

        if n == 1:
            ordered[0].placement_by_grid[grid_id].rank_percentile = 1.0
            continue

        for local_rank, placement in enumerate(ordered):
            placement.placement_by_grid[grid_id].rank_percentile = (
                1.0 - (local_rank / float(n - 1))
            )

# -----------------------------
# Helpers: tile sizing from image
# -----------------------------

def choose_base_sigma_n(tile_size: int,
                        *,
                        base_sigma_n_divisor: int = BASE_SIGMA_N_DIVISOR,
                        base_sigma_n_min: int = BASE_SIGMA_N_PX_MIN,
                        base_sigma_n_max: int = BASE_SIGMA_N_PX_MAX,
) -> int:

    initial_base_sigma_n_px = int(round(tile_size / float(base_sigma_n_divisor)))
    base_sigma_n_px = int(np.clip(initial_base_sigma_n_px, base_sigma_n_min,  base_sigma_n_max, ))
    return  base_sigma_n_px


# -----------------------------
# Tile selection (one grid)
# -----------------------------

def select_grid0_tile_groups(
    placements_by_tile: dict[tuple[int, int, int], list[SeedPlacement]],
) -> dict[tuple[int, int, int], list[SeedPlacement]]:
    """
    Return only tile groups that belong to grid 0.

    Parameters
    ----------
    placements_by_tile
        Tile-grouped placements keyed by:
        (grid_id, tile_row, tile_col)

    Returns
    -------
    grid0_tiles
        Subset of placements_by_tile containing only keys with grid_id == 0.
    """
    return {
        key: placements
        for key, placements in placements_by_tile.items()
        if key[0] == 0
    }

def sort_tile_candidates_for_pairing(
    tile_placements: list[SeedPlacement],
    strength_map: np.ndarray,
    *,
    grid_id: int = 0,
) -> list[SeedPlacement]:
    """
    Return a sorted working list for one tile in one grid.

    Ordering:
      1. larger tile-local rank_percentile first
      2. stronger ridge response first
      3. smaller row
      4. smaller col
    """
    return sorted(
        tile_placements,
        key=lambda p: (
            -float(p.placement_by_grid[grid_id].rank_percentile),
            -float(strength_map[p.row, p.col]),
            p.row,
            p.col,
        ),
    )

def find_candidates_in_partner_gate(
    tile_candidates: list[SeedPlacement],
    *,
    row_gate_center: float,
    col_gate_center: float,
    t_row: float,
    t_col: float,
    n_row: float,
    n_col: float,
    gate_half_width_t_px: float = PARTNER_GATE_HALF_WIDTH_T_PX,
    gate_half_width_n_px: float = PARTNER_GATE_HALF_WIDTH_N_PX,
    exclude_seed_id: int | None = None,
) -> list[SeedPlacement]:
    """
       Return candidates whose positions fall inside the partner gate.

       The gate is centered at (row_gate_center, col_gate_center) and evaluated
       in the local tangent-normal frame defined by (t_row, t_col, n_row, n_col).

       Parameters
       ----------
       tile_candidates
           Candidate placements from one tile worklist.
       row_gate_center, col_gate_center
           Center of the partner gate in image coordinates.
       t_row, t_col
           Tangent unit-vector components of the local frame.
       n_row, n_col
           Normal unit-vector components of the local frame.
       gate_half_width_t_px
           Half-width of the gate along the tangent direction, in pixels.
       gate_half_width_n_px
           Half-width of the gate along the normal direction, in pixels.
       exclude_seed_id
           Optional seed_id to skip, usually the anchor itself.

       Returns
       -------
       in_gate
           Candidates that lie inside the rectangular gate in local-frame coordinates.
       """
    if gate_half_width_t_px < 0.0:
        raise ValueError(
            f"gate_half_width_t_px must be non-negative, got {gate_half_width_t_px}"
        )
    if gate_half_width_n_px < 0.0:
        raise ValueError(
            f"gate_half_width_n_px must be non-negative, got {gate_half_width_n_px}"
        )

    in_gate: list[SeedPlacement] = []

    for candidate in tile_candidates:
        if exclude_seed_id is not None and candidate.seed_id == exclude_seed_id:
            continue

        d_t_gate, d_n_gate = project_point_into_local_frame(
            row_ref=row_gate_center,
            col_ref=col_gate_center,
            t_row=t_row,
            t_col=t_col,
            n_row=n_row,
            n_col=n_col,
            row_test=float(candidate.row),
            col_test=float(candidate.col),
        )

        if abs(d_t_gate) > gate_half_width_t_px:
            continue
        if abs(d_n_gate) > gate_half_width_n_px:
            continue

        in_gate.append(candidate)

    return in_gate

def is_candidate_valid_partner(
    anchor: SeedPlacement,
    candidate: SeedPlacement,
    ridge_map: RidgeMap,
    *,
    expected_side: int,
    partner_theta_tolerance: float = PARTNER_THETA_TOLERANCE,
    partner_rejection_distance_px: float = PARTNER_REJECTION_DISTANCE_PX,
    eps: float = EPS,
) -> bool:
    """
    Return True if candidate is a valid partner for anchor.

    This function assumes the candidate has already passed the partner-gate
    position test. Here we validate:
      - candidate is on the expected side of the anchor normal
      - candidate is not too close to the anchor along the normal
      - candidate tangent agrees with anchor tangent up to axial tolerance
    """
    if expected_side not in (-1, 1):
        raise ValueError(f"expected_side must be -1 or 1, got {expected_side}")

    theta_anchor = float(ridge_map.thetas[anchor.row, anchor.col])
    theta_candidate = float(ridge_map.thetas[candidate.row, candidate.col])

    t_row, t_col, n_row, n_col = local_frame_from_theta(theta_anchor)

    d_t_anchor, d_n_anchor = project_point_into_local_frame(
        row_ref=float(anchor.row),
        col_ref=float(anchor.col),
        t_row=t_row,
        t_col=t_col,
        n_row=n_row,
        n_col=n_col,
        row_test=float(candidate.row),
        col_test=float(candidate.col),
    )

    side = candidate_side_from_normal_projection(d_n_anchor, eps=eps)
    if side == 0:
        return False
    if side != expected_side:
        return False

    if abs(d_n_anchor) < float(partner_rejection_distance_px):
        return False

    theta_diff = axial_angle_difference(theta_anchor, theta_candidate)
    if theta_diff > float(partner_theta_tolerance):
        return False

    return True


def score_candidate_for_anchor(
    anchor: SeedPlacement,
    candidate: SeedPlacement,
    ridge_map: RidgeMap,
    #TODO: not use hardcoded, but derived learned parameters for weight in future
) -> float:
    """
    Return a score for pairing candidate with anchor.

    Lower score is better.

    Score terms:
      - distance from partner-gate center in local tangent direction
      - distance from partner-gate center in local normal direction
      - axial theta mismatch
      - ridge strength reward
    """
    theta_anchor = float(ridge_map.thetas[anchor.row, anchor.col])
    theta_candidate = float(ridge_map.thetas[candidate.row, candidate.col])

    t_row, t_col, n_row, n_col = local_frame_from_theta(theta_anchor)

    row_gate_center, col_gate_center = partner_gate_center(
        row_ref=float(anchor.row),
        col_ref=float(anchor.col),
        n_row=n_row,
        n_col=n_col,
        offset_px=PARTNER_OFFSET_PX,
    )

    d_t_gate, d_n_gate = project_point_into_local_frame(
        row_ref=row_gate_center,
        col_ref=col_gate_center,
        t_row=t_row,
        t_col=t_col,
        n_row=n_row,
        n_col=n_col,
        row_test=float(candidate.row),
        col_test=float(candidate.col),
    )

    d_theta = axial_angle_difference(theta_anchor, theta_candidate)
    candidate_strength = float(ridge_map.vesselness[candidate.row, candidate.col])

    normal_reward = 1.0 - abs(d_n_gate) / PARTNER_GATE_HALF_WIDTH_N_PX
    tangent_reward = 1.0 - abs(d_t_gate) / PARTNER_GATE_HALF_WIDTH_T_PX
    theta_reward = 1.0 - d_theta / PARTNER_THETA_TOLERANCE
    # TODO: use learned weight
    score = (
            GATE_CENTER_N_WEIGHT * normal_reward
            + GATE_CENTER_T_WEIGHT * tangent_reward
            + THETA_WEIGHT * theta_reward
    )

    return float(score)


def find_best_partner_for_anchor(
        anchor: SeedPlacement,
        tile_candidates: list[SeedPlacement],
        ridge_map: RidgeMap,
        *,
        expected_side: int,
        partner_offset_px: float = PARTNER_OFFSET_PX,
        gate_half_width_t_px: float = PARTNER_GATE_HALF_WIDTH_T_PX,
        gate_half_width_n_px: float = PARTNER_GATE_HALF_WIDTH_N_PX,
) -> SeedPlacement | None:
    """
    Return the best valid partner for anchor from one tile worklist.

    The search is performed in three steps:
      1. build the local frame at the anchor
      2. collect candidates inside the partner gate
      3. keep only valid partners and return the highest-scoring one

    Returns
    -------
    best_partner
        Best matching candidate, or None if no valid partner is found.
    """
    theta_anchor = float(ridge_map.thetas[anchor.row, anchor.col])

    t_row, t_col, n_row, n_col = local_frame_from_theta(theta_anchor)

    row_gate_center, col_gate_center = partner_gate_center(
        row_ref=float(anchor.row),
        col_ref=float(anchor.col),
        n_row=n_row,
        n_col=n_col,
        offset_px=float(expected_side) * float(partner_offset_px),
    )

    candidates_in_gate = find_candidates_in_partner_gate(
        tile_candidates,
        row_gate_center=row_gate_center,
        col_gate_center=col_gate_center,
        t_row=t_row,
        t_col=t_col,
        n_row=n_row,
        n_col=n_col,
        gate_half_width_t_px=gate_half_width_t_px,
        gate_half_width_n_px=gate_half_width_n_px,
        exclude_seed_id=anchor.seed_id,
    )

    best_partner: SeedPlacement | None = None
    best_key: tuple[float, float, int, int] | None = None

    for candidate in candidates_in_gate:
        if not is_candidate_valid_partner(
                anchor,
                candidate,
                ridge_map,
                expected_side=expected_side,
        ):
            continue

        score = score_candidate_for_anchor(
            anchor,
            candidate,
            ridge_map,
        )

        candidate_strength = float(ridge_map.vesselness[candidate.row, candidate.col])

        # max() logic:
        #   1. higher score is better
        #   2. higher strength is better
        #   3. smaller row wins ties
        #   4. smaller col wins ties
        candidate_key = (
            float(score),
            candidate_strength,
            -int(candidate.row),
            -int(candidate.col),
        )

        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_partner = candidate

    return best_partner


def make_centerline_seed_from_pair(
    anchor: SeedPlacement,
    partner: SeedPlacement,
    ridge_map: RidgeMap,
    mid_seed_id: int = -1,
) -> CenterlineSeed:
    """
    Build one centerline seed from an anchor-partner pair.

    The center seed is defined by:
      - midpoint position
      - axial-average tangent
      - average strength
      - pair width in pixels
    """
    theta_anchor = float(ridge_map.thetas[anchor.row, anchor.col])
    theta_partner = float(ridge_map.thetas[partner.row, partner.col])

    source_anchor_strength = float(ridge_map.vesselness[anchor.row, anchor.col])
    source_partner_strength = float(ridge_map.vesselness[partner.row, partner.col])

    mid_row = 0.5 * (float(anchor.row) + float(partner.row))
    mid_col = 0.5 * (float(anchor.col) + float(partner.col))

    mid_theta = axial_average_theta(theta_anchor, theta_partner)
    mid_strength = 0.5 * (source_anchor_strength + source_partner_strength)

    mid_width_px = float(
        np.hypot(
            float(partner.row) - float(anchor.row),
            float(partner.col) - float(anchor.col),
        )
    )

    return CenterlineSeed(
        mid_seed_id=int(mid_seed_id),
        mid_row=float(mid_row),
        mid_col=float(mid_col),
        mid_theta=float(mid_theta),
        mid_strength=float(mid_strength),
        mid_width_px=float(mid_width_px),
        source_anchor_seed_id=int(anchor.seed_id),
        source_partner_seed_id=int(partner.seed_id),
        source_anchor_row=int(anchor.row),
        source_anchor_col=int(anchor.col),
        source_partner_row=int(partner.row),
        source_partner_col=int(partner.col),
        source_anchor_strength=float(source_anchor_strength),
        source_partner_strength=float(source_partner_strength),
    )


def build_centerline_seeds_from_grid0_tiles(
    placements_by_tile: dict[tuple[int, int, int], list[SeedPlacement]],
    ridge_map: RidgeMap,
    *,
    grid_id: int = 0,
) -> list[CenterlineSeed]:
    """
    Build centerline seeds from grid-0 tile groups.

    Flow
    ----
    1. keep only grid-0 tile groups
    2. process tiles in deterministic order
    3. greedily pair candidates inside each tile
    4. aggregate all centerline seeds into one flat list

    Parameters
    ----------
    placements_by_tile
        Tile-grouped placements keyed by:
        (grid_id, tile_row, tile_col)
    ridge_map
        Ridge maps used for theta and vesselness lookup.
    grid_id
        Grid to use for pairing. For now this should stay 0.

    Returns
    -------
    mid_seeds
        Flat list of centerline seeds built from all processed grid-0 tiles.
    """
    if grid_id != 0:
        raise ValueError(f"build_centerline_seeds_from_grid0_tiles expects grid_id=0, got {grid_id}")

    grid0_tiles = select_grid0_tile_groups(placements_by_tile)

    ordered_tile_keys = sorted(
        grid0_tiles.keys(),
        key=lambda key: (int(key[1]), int(key[2])),
    )

    mid_seeds: list[CenterlineSeed] = []
    next_mid_seed_id = 0

    for tile_key in ordered_tile_keys:
        tile_candidates = grid0_tiles[tile_key]

        if len(tile_candidates) < 2:
            continue

        tile_mid_seeds, next_mid_seed_id = pair_tile_candidates_into_centerline_seeds(
            tile_candidates,
            ridge_map,
            grid_id=grid_id,
            start_mid_seed_id=next_mid_seed_id,
        )

        mid_seeds.extend(tile_mid_seeds)

    return mid_seeds

def pair_tile_candidates_into_centerline_seeds(
    tile_candidates: list[SeedPlacement],
    ridge_map: RidgeMap,
    *,
    grid_id: int = 0,
    start_mid_seed_id: int = 0,
) -> tuple[list[CenterlineSeed], int]:
    """
    Greedily pair candidates inside one tile and return centerline seeds.

    Behavior
    --------
    - take the best remaining anchor
    - search both partner sides
    - choose the better valid partner
    - if no valid partner is found, drop only the anchor
    - if a partner is found, emit one centerline seed and consume both
    - continue until fewer than 2 candidates remain

    Returns
    -------
    mid_seeds, next_mid_seed_id
        mid_seeds:
            Centerline seeds created from this tile.
        next_mid_seed_id:
            Next free id after processing this tile.
    """
    strength_map = ridge_map.vesselness
    if strength_map is None:
        raise ValueError("ridge_map.vesselness is not available")

    worklist = sort_tile_candidates_for_pairing(
        tile_candidates,
        strength_map,
        grid_id=grid_id,
    )

    mid_seeds: list[CenterlineSeed] = []
    next_mid_seed_id = int(start_mid_seed_id)

    while len(worklist) >= 2:
        anchor = worklist[0]

        partner_pos = find_best_partner_for_anchor(
            anchor,
            worklist,
            ridge_map,
            expected_side=1,
        )

        partner_neg = find_best_partner_for_anchor(
            anchor,
            worklist,
            ridge_map,
            expected_side=-1,
        )

        chosen_partner: SeedPlacement | None = None

        if partner_pos is None and partner_neg is None:
            # No valid partner -> drop only the anchor.
            worklist.pop(0)
            continue

        if partner_pos is None:
            chosen_partner = partner_neg
        elif partner_neg is None:
            chosen_partner = partner_pos
        else:
            score_pos = score_candidate_for_anchor(anchor, partner_pos, ridge_map)
            score_neg = score_candidate_for_anchor(anchor, partner_neg, ridge_map)

            if score_pos > score_neg:
                chosen_partner = partner_pos
            elif score_neg > score_pos:
                chosen_partner = partner_neg
            else:
                # Deterministic tie-break:
                # stronger partner first, then smaller row, then smaller col
                strength_pos = float(ridge_map.vesselness[partner_pos.row, partner_pos.col])
                strength_neg = float(ridge_map.vesselness[partner_neg.row, partner_neg.col])

                key_pos = (strength_pos, -int(partner_pos.row), -int(partner_pos.col))
                key_neg = (strength_neg, -int(partner_neg.row), -int(partner_neg.col))

                chosen_partner = partner_pos if key_pos >= key_neg else partner_neg

        if chosen_partner is None:
            worklist.pop(0)
            continue

        mid_seed = make_centerline_seed_from_pair(
            anchor=anchor,
            partner=chosen_partner,
            ridge_map=ridge_map,
        )

        # Temporary id assignment happens here.
        mid_seed.mid_seed_id = next_mid_seed_id
        next_mid_seed_id += 1

        mid_seeds.append(mid_seed)

        consumed_seed_ids = {int(anchor.seed_id), int(chosen_partner.seed_id)}
        worklist = [
            candidate
            for candidate in worklist
            if int(candidate.seed_id) not in consumed_seed_ids
        ]

    return mid_seeds, next_mid_seed_id


def prune_tile_placements(
    placements: list[SeedPlacement],
    placements_by_tile: dict[tuple[int, int, int], list[SeedPlacement]],
    strength_map: np.ndarray,
    *,
    seeds_per_tile: int = SEEDS_PER_TILE,
) -> list[SeedPlacement]:
    #TODO: get tid of triple dict
    """
    Keep at most seeds_per_tile grid-memberships per tile.

    The same physical seed may survive in one grid and be removed from another.
    """
    surviving_memberships: set[tuple[int, int]] = set()
    # (seed_id, grid_id)

    for (grid_id, _, _), tile_placements in placements_by_tile.items():
        if not tile_placements:
            continue

        ordered = sorted(
            tile_placements,
            key=lambda p: (
                -float(p.placement_by_grid[grid_id].rank_percentile),
                -float(strength_map[p.row, p.col]),
                p.row,
                p.col,
            ),
        )

        winners = ordered[:seeds_per_tile]

        for placement in winners:
            surviving_memberships.add((placement.seed_id, grid_id))

    surviving_placements: list[SeedPlacement] = []

    for placement in placements:
        for grid_id in list(placement.placement_by_grid.keys()):
            if (placement.seed_id, grid_id) not in surviving_memberships:
                placement.placement_by_grid.pop(grid_id, None)

        if placement.placement_by_grid:
            surviving_placements.append(placement)

    return surviving_placements


# -----------------------------
# Materialize seeds
# -----------------------------

def materialize_seeds_from_placements(
    placements: Iterable[SeedPlacement],
    ridge_map: RidgeMap,
    global_rank_map: np.ndarray,
    *,
    base_sigma_n_px: float,
    hessian_scale_px: float = HESSIAN_SCALE_PX,
    flat_area_cap: float = FLAT_AREA_CAP,
    eps: float = EPS,
) -> list[Seed]:
    """
    Build one physical Seed per surviving SeedPlacement.

    Notes
    -----
    - Expects placements after prune_tile_placements.
    - Ignores placements whose placement_by_grid is empty.
    - Uses global_rank_map for Seed.rank_percentile.
    - Keeps tile-local percentiles only inside placement_by_grid.
    """
    seeds: list[Seed] = []
    seen_seed_ids: set[int] = set()

    for placement in placements:
        if not placement.placement_by_grid:
            continue

        if placement.seed_id in seen_seed_ids:
            raise ValueError(f"Duplicate surviving seed_id={placement.seed_id}")
        seen_seed_ids.add(placement.seed_id)

        row = int(placement.row)
        col = int(placement.col)

        strength = float(ridge_map.vesselness[row, col])
        rank_percentile = float(global_rank_map[row, col])
        theta = float(ridge_map.thetas[row, col])
        lambda_perp = float(ridge_map.lambdas_perp[row, col])
        lambda_para = float(ridge_map.lambdas_par[row, col])
        eigen_anisotropy = float(ridge_map.anisotropies[row, col])

        sigma_n_px, sigma_t_px = derive_sigmas_from_local_geometry(
            base_scale_px=float(base_sigma_n_px),
            eigen_anisotropy=eigen_anisotropy,
            theta=theta,
            flat_area_cap=float(flat_area_cap),
            eps=float(eps),
        )

        seeds.append(
            Seed(
                seed_id=int(placement.seed_id),
                row=row,
                col=col,
                strength=strength,
                rank_percentile=rank_percentile,
                theta=theta,
                sigma_n_px=float(sigma_n_px),
                sigma_t_px=float(sigma_t_px),
                hessian_scale_px=float(hessian_scale_px),
                lambda_perp=lambda_perp,
                lambda_para=lambda_para,
                eigen_anisotropy=eigen_anisotropy,
            )
        )

    return seeds




def suppress_seeds_with_anisotropic_distance(
    seeds: Iterable[Seed],
    *,
    suppression_distance_threshold: float,
    max_seeds: int | None = None,
    support_view: str = "all",
) -> tuple[list[Seed], list[SuppressionRecord]]:
    """
    Strength-sorted greedy suppression in normalized anisotropic distance.

    Keep a seed only if its anisotropic distance from every already-kept seed
    is at least suppression_distance_threshold.

    Returns
    -------
    kept, records
        kept    : surviving seeds after global suppression
        records : one SuppressionRecord per processed seed
    """
    seeds = list(seeds)

    if suppression_distance_threshold < 0:
        raise ValueError(
            f"suppression_distance_threshold must be non-negative, "
            f"got {suppression_distance_threshold}"
        )

    suppression_distance_threshold_sq = (
        float(suppression_distance_threshold) * float(suppression_distance_threshold)
    )

    # strongest first, then deterministic tie-breaks
    seeds.sort(key=lambda s: (-s.strength, -s.rank_percentile, s.row, s.col))

    kept: list[Seed] = []
    records: list[SuppressionRecord] = []

    for seed in seeds:
        suppressed = False
        num_compared_kept_seeds = 0

        nearest_compared_seed_id: int | None = None
        nearest_compared_distance2: float | None = None

        blocking_seed_id: int | None = None
        blocking_distance2: float | None = None

        for kept_seed in kept:
            num_compared_kept_seeds += 1

            distance2 = anisotropic_distance2(
                row_ref=kept_seed.row,
                col_ref=kept_seed.col,
                theta_ref=kept_seed.theta,
                row_test=seed.row,
                col_test=seed.col,
                sigma_n_px=kept_seed.sigma_n_px,
                sigma_t_px=kept_seed.sigma_t_px,
            )

            if nearest_compared_distance2 is None or distance2 < nearest_compared_distance2:
                nearest_compared_distance2 = float(distance2)
                nearest_compared_seed_id = int(kept_seed.seed_id)

            if distance2 < suppression_distance_threshold_sq:
                suppressed = True
                blocking_seed_id = int(kept_seed.seed_id)
                blocking_distance2 = float(distance2)
                break

        if suppressed:
            records.append(
                make_suppression_record(
                    seed,
                    support_view=support_view,
                    suppression_distance_threshold=suppression_distance_threshold,
                    decision="suppressed",
                    kept_index=None,
                    num_compared_kept_seeds=num_compared_kept_seeds,
                    nearest_compared_seed_id=nearest_compared_seed_id,
                    nearest_compared_distance2=nearest_compared_distance2,
                    blocking_seed_id=blocking_seed_id,
                    blocking_distance2=blocking_distance2,
                )
            )
            continue

        kept_index = len(kept)
        kept.append(seed)

        records.append(
            make_suppression_record(
                seed,
                support_view=support_view,
                suppression_distance_threshold=suppression_distance_threshold,
                decision="kept",
                kept_index=kept_index,
                num_compared_kept_seeds=num_compared_kept_seeds,
                nearest_compared_seed_id=nearest_compared_seed_id,
                nearest_compared_distance2=nearest_compared_distance2,
                blocking_seed_id=None,
                blocking_distance2=None,
            )
        )

        if max_seeds is not None and len(kept) >= max_seeds:
            break

    return kept, records



# -----------------------------
#  Diagnostic Functions --> check different grids to decide if its needed
# -----------------------------

from typing import Iterable

def snapshot_survived_in_grids_in_place(
    placements: Iterable[SeedPlacement],
) -> None:
    for placement in placements:
        placement.survived_in_grids = sorted(placement.placement_by_grid.keys())


def split_materialized_seeds_by_grid_support(
    placements: Iterable[SeedPlacement],
    seeds: Iterable[Seed],
) -> tuple[list[Seed], list[Seed], list[Seed]]:
    """
    Returns:
        grid0_seeds, grid1_seeds, both_grids_seeds
    """
    support_by_seed_id: dict[int, tuple[int, ...]] = {
        int(p.seed_id): tuple(sorted(p.survived_in_grids))
        for p in placements
        if p.placement_by_grid
    }

    seeds_by_id: dict[int, Seed] = {int(s.seed_id): s for s in seeds}

    grid0_seeds: list[Seed] = []
    grid1_seeds: list[Seed] = []
    both_grids_seeds: list[Seed] = []

    for seed_id, grids in support_by_seed_id.items():
        seed = seeds_by_id.get(seed_id)
        if seed is None:
            continue

        if 0 in grids:
            grid0_seeds.append(seed)

        if 1 in grids:
            grid1_seeds.append(seed)

        if tuple(grids) == (0, 1):
            both_grids_seeds.append(seed)

    return grid0_seeds, grid1_seeds, both_grids_seeds


def suppress_seeds_by_grid_support(
    placements: Iterable[SeedPlacement],
    seeds: Iterable[Seed],
    *,
    suppression_distance_threshold: float,
    max_seeds: int | None = None,
) -> tuple[dict[str, list[Seed]], dict[str, list[SuppressionRecord]]]:
    """
    Diagnostic suppression views after prune/materialization.

    Returns
    -------
    kept_by_view, records_by_view
        kept_by_view keys:
            - "grid0"
            - "grid1"
            - "both"

        records_by_view keys:
            - "grid0"
            - "grid1"
            - "both"
    """
    placements = list(placements)
    seeds = list(seeds)

    snapshot_survived_in_grids_in_place(placements)

    grid0_seeds, grid1_seeds, both_grids_seeds = split_materialized_seeds_by_grid_support(
        placements,
        seeds,
    )

    grid0_kept, grid0_records = suppress_seeds_with_anisotropic_distance(
        grid0_seeds,
        suppression_distance_threshold=suppression_distance_threshold,
        max_seeds=max_seeds,
        support_view="grid0",
    )

    grid1_kept, grid1_records = suppress_seeds_with_anisotropic_distance(
        grid1_seeds,
        suppression_distance_threshold=suppression_distance_threshold,
        max_seeds=max_seeds,
        support_view="grid1",
    )

    both_kept, both_records = suppress_seeds_with_anisotropic_distance(
        both_grids_seeds,
        suppression_distance_threshold=suppression_distance_threshold,
        max_seeds=max_seeds,
        support_view="both",
    )

    kept_by_view = {
        "grid0": grid0_kept,
        "grid1": grid1_kept,
        "both": both_kept,
    }

    records_by_view = {
        "grid0": grid0_records,
        "grid1": grid1_records,
        "both": both_records,
    }

    return kept_by_view, records_by_view
def build_seeding_debug_info(
    src_img: np.ndarray,
    *,
    q: float,
    tile_size: int,
    base_sigma_n_px: float,
    seeds_per_tile: int,
    grids: tuple[Grid, Grid],
    seeds_mask: np.ndarray,
    placements_before_prune: list[SeedPlacement],
    placements_after_prune: list[SeedPlacement],
    seeds_before_suppression: list[Seed],
    suppressed_grid0: list[Seed],
    suppressed_grid1: list[Seed],
    suppressed_both: list[Seed],
    suppression_distance_threshold: float,
    max_seeds: int | None,
) -> dict:
    rows, cols = src_img.shape[:2]
    grid0, grid1 = grids

    return {
        "rows": int(rows),
        "cols": int(cols),
        "q": float(q),
        "tile_size": int(tile_size),
        "base_sigma_n_px": float(base_sigma_n_px),
        "seeds_per_tile": int(seeds_per_tile),
        "grid0_offset": (int(grid0.offset_row), int(grid0.offset_col)),
        "grid1_offset": (int(grid1.offset_row), int(grid1.offset_col)),
        "num_strong_pixels": int(seeds_mask.sum()),
        "num_seed_placements_before_prune": int(len(placements_before_prune)),
        "num_seed_placements_after_prune": int(len(placements_after_prune)),
        "num_seeds_before_suppression": int(len(seeds_before_suppression)),
        "num_seeds_after_suppression_grid0": int(len(suppressed_grid0)),
        "num_seeds_after_suppression_grid1": int(len(suppressed_grid1)),
        "num_seeds_after_suppression_both": int(len(suppressed_both)),
        "max_seeds": None if max_seeds is None else int(max_seeds),
        "suppression_distance_threshold": float(suppression_distance_threshold),
    }

def make_suppression_record(
    seed: Seed,
    *,
    support_view: str,                      #"grid0", "grid1", "both", or "all"
    suppression_distance_threshold: float,
    decision: Literal["kept", "suppressed"],
    kept_index: int | None,
    num_compared_kept_seeds: int,
    nearest_compared_seed_id: int | None,
    nearest_compared_distance2: float | None,
    blocking_seed_id: int | None,
    blocking_distance2: float | None,
) -> SuppressionRecord:
    nearest_compared_distance = (
        None
        if nearest_compared_distance2 is None
        else math.sqrt(nearest_compared_distance2)
    )

    blocking_distance = (
        None
        if blocking_distance2 is None
        else math.sqrt(blocking_distance2)
    )

    suppression_distance_threshold_sq = (
        float(suppression_distance_threshold) * float(suppression_distance_threshold)
    )

    return SuppressionRecord(
        # --- copied from Seed ---
        seed_id=seed.seed_id,
        row=seed.row,
        col=seed.col,
        strength=seed.strength,
        rank_percentile=seed.rank_percentile,
        theta=seed.theta,
        sigma_n_px=seed.sigma_n_px,
        sigma_t_px=seed.sigma_t_px,
        hessian_scale_px=seed.hessian_scale_px,
        lambda_perp=seed.lambda_perp,
        lambda_para=seed.lambda_para,
        eigen_anisotropy=seed.eigen_anisotropy,

        # --- run/view context ---
        support_view=support_view,
        suppression_distance_threshold=float(suppression_distance_threshold),
        suppression_distance_threshold_sq=suppression_distance_threshold_sq,

        # --- suppression outcome ---
        decision=decision,
        kept_index=kept_index,
        num_compared_kept_seeds=num_compared_kept_seeds,

        # --- nearest among actually compared kept seeds ---
        nearest_compared_seed_id=nearest_compared_seed_id,
        nearest_compared_distance2=nearest_compared_distance2,
        nearest_compared_distance=nearest_compared_distance,

        # --- actual blocker ---
        blocking_seed_id=blocking_seed_id,
        blocking_distance2=blocking_distance2,
        blocking_distance=blocking_distance,
    )