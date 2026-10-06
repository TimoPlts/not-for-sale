"""Research tools: parameter sweeps, walk-forward evaluation and agent attribution."""

from trading_lab.research.attribution import (
    Attribution,
    CalibrationBucket,
    attribute,
    attribute_result,
    attribute_run,
    format_attribution,
)
from trading_lab.research.sweep import (
    MemoizedProvider,
    SweepResult,
    apply_params,
    expand_grid,
    rank_key,
    run_sweep,
)
from trading_lab.research.walkforward import (
    WalkForwardFold,
    WalkForwardResult,
    make_folds,
    walk_forward,
)

__all__ = [
    "Attribution",
    "CalibrationBucket",
    "attribute",
    "attribute_result",
    "attribute_run",
    "format_attribution",
    "MemoizedProvider",
    "SweepResult",
    "WalkForwardFold",
    "WalkForwardResult",
    "apply_params",
    "expand_grid",
    "make_folds",
    "rank_key",
    "run_sweep",
    "walk_forward",
]
