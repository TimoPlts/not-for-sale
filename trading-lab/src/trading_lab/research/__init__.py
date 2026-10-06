"""Research tools: parameter sweeps, walk-forward evaluation and agent attribution."""

from trading_lab.research.attribution import (
    Attribution,
    CalibrationBucket,
    attribute,
    attribute_result,
    attribute_run,
    format_attribution,
)
from trading_lab.research.experiments import (
    AGENT_STRATEGIES,
    DEFAULT_VARIANTS,
    VARIANTS,
    ExperimentRow,
    run_experiment,
    variant_config,
    variant_overrides,
)
from trading_lab.research.protocol import VariantSummary, sign_test_p, summarize
from trading_lab.research.sweep import (
    MemoizedProvider,
    SweepResult,
    apply_params,
    expand_grid,
    rank_key,
    run_sweep,
)
from trading_lab.research.weighting import WeightingRule, adaptive_weights
from trading_lab.research.walkforward import (
    WalkForwardFold,
    WalkForwardResult,
    make_folds,
    walk_forward,
)

__all__ = [
    "WeightingRule",
    "adaptive_weights",
    "VariantSummary",
    "sign_test_p",
    "summarize",
    "AGENT_STRATEGIES",
    "DEFAULT_VARIANTS",
    "VARIANTS",
    "ExperimentRow",
    "run_experiment",
    "variant_config",
    "variant_overrides",
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
