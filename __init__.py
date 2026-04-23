"""
xtrendll — learnable-lag (DeltaLag-style) extension of XTrendCS.

Lead-lag information enters the model at two places; neither touches the
dataset:
  * peer_h is sliced at past time offsets inside LagAwarePeerBlock,
  * Bennett's static adjacency matrix S (optional, A2-only) is loaded as
    a non-trainable buffer on the block.

The package mirrors the existing project layout; no files outside this
folder are modified.
"""

from .lag_blocks import LagAwarePeerBlock
from .x_trend_ll import (
    XTrendLL, LL_DEFAULT,
    _xtrendll_step, _xtrendll_step_panel,
    make_xtrendll_step, make_xtrendll_step_panel,
)
from .lead_lag import (
    build_lead_lag_matrix, build_lead_lag_matrix_cached, align_S_to_asset_ids,
    build_lag_ranking_artifact, build_lag_ranking_cached,
    artifact_to_lag_strength_tensor, artifact_to_lag_topk_mask_tensor,
)
from .prep_artifacts import load_artifacts
from .results_io import save_run, load_run, compare_runs

__all__ = [
    "LagAwarePeerBlock",
    "XTrendLL",
    "LL_DEFAULT",
    "_xtrendll_step",
    "_xtrendll_step_panel",
    "make_xtrendll_step",
    "make_xtrendll_step_panel",
    "build_lead_lag_matrix",
    "build_lead_lag_matrix_cached",
    "align_S_to_asset_ids",
    "build_lag_ranking_artifact",
    "build_lag_ranking_cached",
    "artifact_to_lag_strength_tensor",
    "artifact_to_lag_topk_mask_tensor",
    "load_artifacts",
    "save_run",
    "load_run",
    "compare_runs",
]
