# src/pipeline.py

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from skimage import measure


from configs.config import (
    DATA_DIR,
    RESULTS_DIR,
    RESULTS_VIS_DIR,
    DEFAULT_SEGMENTATION_METHOD,
    DEFAULT_PREFIX,
    LOCAL_NORM_SIGMA,
    GAUSSIAN_SMOOTH_SIGMA, QUANTILE, MAX_SEEDS, SEEDS_PER_TILE, HESSIAN_SCALE_PX,
    SUPPRESSION_DISTANCE_THRESHOLD,
    ANCHOR_COLOR,
    MIDPOINT_COLOR,
    PARTNER_COLOR
)

from .utilities import (
    load_image,
    collect_images_paths,
    ensure_dir,
    rgba_from_gray,
    save_rgba_tiff_from_gray,
    save_processed_parameters,
    save_outputs_with_metadata,
    SAVE_NPY,
    SAVE_TIFF2D_F32,
    SAVE_RGBA_TIFF,
)
#from .preprocessing import preprocess_image
from .preprocessing import (
    to_gray_normalized,
    local_normalize_HOG_style,
    local_contrast_normalization_CLAHE,
    gaussian_smoothing
)

from .ridges import RidgeMap
from .import ridge_seeding as rs
from .import visualization as viz
from .debug_io import save_rgba_tiff

from .segmentation import run_segmentation


SegMethod = Literal["otsu", "sam"]


class DislocationPipeline:
    """
    High-level pipeline for:
    1) Load
    2) Preprocess
    3) Segment
    4) Measure regions
    5) Save outputs (+ JSON metadata)
    """

    def __init__(
        self,
        input_paths: list[str | Path] | None = None,
        input_dir: str | Path | None = DATA_DIR,
        segmentation_method: SegMethod | None = None,
        prefix: str | None = None,
        save_vis: bool = False,
        max_files: int | None = 1
    ) -> None:


        self.segmentation_method: SegMethod = (segmentation_method or DEFAULT_SEGMENTATION_METHOD)  # type: ignore
        self.prefix = prefix or DEFAULT_PREFIX
        self.save_vis = save_vis
        # TODO: look very closely to prefix and think about it
        # keep “results/<prefix>” for CSV outputs etc.
        self.results_dir = ensure_dir(Path(RESULTS_DIR) / self.prefix)

        if input_paths is not None:
            self.input_paths = [Path(p) for p in input_paths]
        elif input_dir is not None:
            self.input_paths = collect_images_paths(Path(input_dir))
        else:
            self.input_paths = [DATA_DIR / "Figure_10.png"]


    def run(self) -> list[dict]:
        return [self._run_single(p) for p in self.input_paths]

    def run_one(self) -> dict:
        """
        Run the pipeline for exactly one input image.
        Convenience wrapper for single-image workflows.
        """
        if len(self.input_paths) != 1:
            raise RuntimeError(f"run_one() requires exactly one input image, got {len(self.input_paths)}")
        return self._run_single(self.input_paths[0])

    def _run_single(self, input_path: Path) -> dict:
        # 1) Load raw

        img_raw = load_image(input_path)

        # 2) Preprocess
#        img_pre = preprocess_image(
#            img_raw,
#            input_path,
#            False,
#        )

        gray = to_gray_normalized(img_raw)

        pre_steps = [
            local_normalize_HOG_style  , #,local_contrast_normalization_CLAHE
            gaussian_smoothing,
        ]

        img_pre = gray
        for fn in pre_steps:
            img_pre = fn(img_pre)

        # Save preprocessed outputs (+ METADATA)
        pre_cfg = {
            "stage": "preprocess",
            "prefix": self.prefix,
            "preprocess_version": "v1_hog_lcn_gauss",
            # put ONLY parameters that define the result here:
            "steps": [fn.__name__ for  fn in pre_steps ],
            "params": {
                "LOCAL_NORM_SIGMA": LOCAL_NORM_SIGMA,
                "GAUSSIAN_SMOOTH_SIGMA": GAUSSIAN_SMOOTH_SIGMA,
            },

        }
        pre_modes = [SAVE_NPY, SAVE_TIFF2D_F32] + ([SAVE_RGBA_TIFF] if self.save_vis else [])
        pre_out = save_outputs_with_metadata(
            image2d=img_pre,
            input_path=input_path,
            config=pre_cfg,
            modes=pre_modes,
            name_override=f"{self.prefix}_pre",
        )

        # 2.5) Ridges + Seeding (NEW)
        ridge_map = RidgeMap(img_pre)
        ridge_out = ridge_map.vesselness

        seeds_mask = rs.select_strong_positions(ridge_out)
        global_ranks_1d = rs.rank_strong_positions(ridge_out, seeds_mask)
        global_rank_map = rs.map_ranked_positions(seeds_mask, global_ranks_1d)

        tile_size = rs.choose_tile_size(ridge_out)
        grids = rs.define_grids(tile_size)

        placements  = rs.create_seed_placements(seeds_mask, grids)
        placements_by_tile = rs.group_seed_placements_by_tile(placements )
        rs.assign_local_rank_percentiles_in_place(
            placements_by_tile,
            ridge_out,
            global_rank_map,
        )

        centerline_seeds = rs.build_centerline_seeds_from_grid0_tiles(
            placements_by_tile,
            ridge_map,
        )

        centerline_csv_path = self.results_dir / f"{self.prefix}_centerline_seeds.csv"
        centerline_df = pd.DataFrame.from_records([vars(s) for s in centerline_seeds])
        centerline_df.to_csv(centerline_csv_path, index=False)

        # --- Visualisation ---
        centerline_vis_path = self.results_dir / f"{self.prefix}_centerline_seeds.tiff"
        base_rgba = rgba_from_gray(img_pre)

        viz.paint_points_in_place(
            base_rgba,
            [s.source_anchor_row for s in centerline_seeds],
            [s.source_anchor_col for s in centerline_seeds],
            color=ANCHOR_COLOR,
            half_width=1,
        )

        viz.paint_points_in_place(
            base_rgba,
            [s.source_partner_row for s in centerline_seeds],
            [s.source_partner_col for s in centerline_seeds],
            color=PARTNER_COLOR,
            half_width=1,
        )

        viz.paint_points_in_place(
            base_rgba,
            [s.mid_row for s in centerline_seeds],
            [s.mid_col for s in centerline_seeds],
            color=MIDPOINT_COLOR,
            half_width=2,
            round_coords=True,
        )

        save_rgba_tiff(base_rgba, centerline_vis_path)

        ridge_debug = {
            "rows": int(img_pre.shape[0]),
            "cols": int(img_pre.shape[1]),
            "q": float(QUANTILE),
            "tile_size": int(tile_size),
            "num_strong_pixels": int(seeds_mask.sum()),
            "num_seed_placements": int(len(placements)),
            "num_tile_groups": int(len(placements_by_tile)),
            "num_centerline_seeds": int(len(centerline_seeds)),
            "centerline_csv_path": str(centerline_csv_path),
            "centerline_vis_path": str(centerline_vis_path),
        }

        ridge_cfg = {
            "stage": "ridge_centerline",
            "centerline_seeds_path": str(centerline_csv_path),
            "num_centerline_seeds": int(len(centerline_seeds)),
            "q": float(QUANTILE),
            "hessian_scale_px": float(HESSIAN_SCALE_PX),
            "debug": ridge_debug,
        }

        ridge_saved = save_outputs_with_metadata(
            image2d=ridge_out,
            input_path=input_path,
            config=ridge_cfg,
            modes=[SAVE_NPY, SAVE_TIFF2D_F32] + ([SAVE_RGBA_TIFF] if self.save_vis else []),
            name_override=f"{self.prefix}_V",
        )




        #TODO: Fix this shit





        # 3) Segmentation
        binary, labels_ws = run_segmentation(
            img_pre,
            input_path,
            method=self.segmentation_method,
            min_size=20,  # same reason: we handle saving centrally now
            save_outputs=False,
        )

        # Save segmentation outputs (+ metadata)
        seg_cfg = {
            "stage": "segmentation",
            "prefix": self.prefix,
            "method": self.segmentation_method,
            # include algorithm parameters that affect output:
            "otsu_min_size": 20,
            "watershed": {"footprint": [3, 3], "exclude_border": False},
        }
        seg_save_modes = [SAVE_NPY, SAVE_TIFF2D_F32] + ([SAVE_RGBA_TIFF] if self.save_vis else [])

        bin_out = save_outputs_with_metadata(
            image2d=binary.astype(np.float32),
            input_path=input_path,

            
            config=seg_cfg | {"output": "binary"},
            modes=seg_save_modes,
            name_override=f"{self.prefix}_binary",
        )
        lbl_out = save_outputs_with_metadata(
            image2d=labels_ws.astype(np.float32),
            input_path=input_path,
            config=seg_cfg | {"output": "labels_ws"},
            modes=seg_save_modes,
            name_override=f"{self.prefix}_labels_ws",
        )

        # 4) Measure regions and save CSV
        params_df = self._measure_regions(labels_ws)
        csv_path = self.results_dir / f"{self.prefix}_loop_parameters.csv"
        save_processed_parameters(params_df, csv_path)

        return {
            "input_path": input_path,
            "img_raw": img_raw,
            "img_pre": img_pre,
            "centerline_seeds": centerline_seeds,
            "binary": binary,
            "labels_ws": labels_ws,
            "loop_params": params_df,
            "saved": {
                "pre": pre_out,
                "ridge": ridge_saved,
                "centerline_csv": centerline_csv_path,
                "centerline_vis": centerline_vis_path,
                "binary": bin_out,
                "labels_ws": lbl_out,
                "csv": csv_path,
            },
        }

    def _measure_regions(self, labels_ws: np.ndarray) -> pd.DataFrame:
        props = measure.regionprops(labels_ws)

        records: list[dict] = []
        for region_id, region in enumerate(props, start=1):
            cy, cx = region.centroid
            major = getattr(region, "major_axis_length", np.nan)
            minor = getattr(region, "minor_axis_length", np.nan)
            orient_rad = getattr(region, "orientation", 0.0)
            orient_deg = float(orient_rad * 180.0 / np.pi)
            area = region.area

            records.append(
                {
                    "id": region_id,
                    "center_x": cx,
                    "center_y": cy,
                    "major_axis": major,
                    "minor_axis": minor,
                    "orientation_deg": orient_deg,
                    "area_px": area,
                    "is_overlapping": False,
                    "is_concentric": False,
                    "id_overlapping": -1,
                    "id_concentric": -1,
                }
            )

        return pd.DataFrame.from_records(records)


def run_default_pipeline() -> dict: # list[dict]:
    pipeline = DislocationPipeline()
    return pipeline.run_one()
"""

JUNK FROM BEFORE



        placements_after_prune = rs.prune_tile_placements(
            placements_before_prune,
            placements_by_tile,
            ridge_out,
        )

        base_sigma_n_px = rs.choose_base_sigma_n(tile_size)

        seeds = rs.materialize_seeds_from_placements(
            placements_after_prune,
            ridge_map,
            global_rank_map,
            base_sigma_n_px=base_sigma_n_px,
        )

        suppressed_by_view, suppression_records_by_view = rs.suppress_seeds_by_grid_support(
            placements_after_prune,
            seeds,
            suppression_distance_threshold=SUPPRESSION_DISTANCE_THRESHOLD,
            max_seeds=MAX_SEEDS,
        )

        suppressed_grid0 = suppressed_by_view["grid0"]
        suppressed_grid1 = suppressed_by_view["grid1"]
        suppressed_both = suppressed_by_view["both"]

        grid0_records = suppression_records_by_view["grid0"]
        grid1_records = suppression_records_by_view["grid1"]
        both_records = suppression_records_by_view["both"]

        # Visualisation

        suppressed_views = viz.visualize_suppressed_grid_views(
            img_pre,
            grid0_seeds=suppressed_grid0,
            grid1_seeds=suppressed_grid1,
            both_grid_seeds=suppressed_both,
            half_width=2,
            alpha_u8=160,
        )

        grid0_vis_path = self.results_dir / f"{self.prefix}_suppressed_grid0.tiff"
        grid1_vis_path = self.results_dir / f"{self.prefix}_suppressed_grid1.tiff"
        both_vis_path = self.results_dir / f"{self.prefix}_suppressed_both.tiff"
        combined_vis_path = self.results_dir / f"{self.prefix}_suppressed_grid01_combined.tiff"

        save_rgba_tiff(suppressed_views["grid0_only_view"], grid0_vis_path)
        save_rgba_tiff(suppressed_views["grid1_only_view"], grid1_vis_path)
        save_rgba_tiff(suppressed_views["both_only_view"], both_vis_path)
        save_rgba_tiff(suppressed_views["combined_grid01_view"], combined_vis_path)
        
        
        # --- save physical seeds only ---
        seeds_csv_path = self.results_dir / f"{self.prefix}_seeds.csv"
        seeds_df = pd.DataFrame.from_records(
            [
                {
                    "seed_id": s.seed_id,
                    "row": s.row,
                    "col": s.col,
                    "strength": s.strength,
                    "rank_percentile": s.rank_percentile,
                    "theta": s.theta,
                    "sigma_n_px": s.sigma_n_px,
                    "sigma_t_px": s.sigma_t_px,
                    "hessian_scale_px": s.hessian_scale_px,
                    "lambda_perp": s.lambda_perp,
                    "lambda_para": s.lambda_para,
                    "eigen_anisotropy": s.eigen_anisotropy,
                }
                for s in seeds
            ]
        )
        seeds_df.to_csv(seeds_csv_path, index=False)

        # --- save suppression records ---
        grid0_records_csv_path = self.results_dir / f"{self.prefix}_suppression_grid0.csv"
        grid1_records_csv_path = self.results_dir / f"{self.prefix}_suppression_grid1.csv"
        both_records_csv_path = self.results_dir / f"{self.prefix}_suppression_both.csv"

        pd.DataFrame.from_records([vars(r) for r in grid0_records]).to_csv(grid0_records_csv_path, index=False)
        pd.DataFrame.from_records([vars(r) for r in grid1_records]).to_csv(grid1_records_csv_path, index=False)
        pd.DataFrame.from_records([vars(r) for r in both_records]).to_csv(both_records_csv_path, index=False)

        seed_debug = {
            "rows": int(img_pre.shape[0]),
            "cols": int(img_pre.shape[1]),
            "q": float(QUANTILE),
            "tile_size": int(tile_size),
            "base_sigma_n_px": float(base_sigma_n_px),
            "seeds_per_tile": int(SEEDS_PER_TILE),
            "num_strong_pixels": int(seeds_mask.sum()),
            "num_seed_placements_before_prune": int(len(placements_before_prune)),
            "num_seed_placements_after_prune": int(len(placements_after_prune)),
            "num_seeds_before_suppression": int(len(seeds)),
            "num_seeds_after_suppression_grid0": int(len(suppressed_grid0)),
            "num_seeds_after_suppression_grid1": int(len(suppressed_grid1)),
            "num_seeds_after_suppression_both": int(len(suppressed_both)),
            "suppression_distance_threshold": float(SUPPRESSION_DISTANCE_THRESHOLD),
            "suppression_record_csvs": {
                "grid0": str(grid0_records_csv_path),
                "grid1": str(grid1_records_csv_path),
                "both": str(both_records_csv_path),
            },
            "visualizations": {
                "grid0_only_view": str(grid0_vis_path),
                "grid1_only_view": str(grid1_vis_path),
                "both_only_view": str(both_vis_path),
                "combined_grid01_view": str(combined_vis_path),
            },
        }

        seed_cfg = {
            "stage": "ridge",
            "seeds_path": str(seeds_csv_path),
            "num_seeds": int(len(seeds)),
            "q": float(QUANTILE),
            "max_seeds": int(MAX_SEEDS),
            "seeds_per_tile": int(SEEDS_PER_TILE),
            "hessian_scale_px": float(HESSIAN_SCALE_PX),
            "suppression_distance_threshold": float(SUPPRESSION_DISTANCE_THRESHOLD),
            "debug": seed_debug,
        }
        
        
        # TODO: fix save_outputs_with_metadata to not necessary save image and modes=None to not do the default
        ridge_saved  = save_outputs_with_metadata(
            image2d=ridge_out,  # 2D -> passes the check
            input_path=input_path,
            config=seed_cfg,
            modes=None,
            name_override=f"{self.prefix}_V",
        )

"""