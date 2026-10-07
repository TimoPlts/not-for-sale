"""Research tools: parameter sweeps, walk-forward evaluation and agent attribution."""

from trading_lab.research.ab import ABResult, ab_test, config_diff, format_ab
from trading_lab.research.agent_eval import AgentEval, evaluate_agents, evaluate_run, format_agent_eval
from trading_lab.research.costs import CostSensitivity, cost_sensitivity, format_costs
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
from trading_lab.research.permutation import PermutationResult, format_permutation, permutation_test
from trading_lab.research.protocol import VariantSummary, sign_test_p, summarize
from trading_lab.research.regimes import RegimeReport, format_regimes, regimes_for_run
from trading_lab.research.reconcile import Reconciliation, format_reconciliation, reconcile
from trading_lab.research.robustness import Robustness, bootstrap, format_robustness, robustness_for_run
from trading_lab.research.sweep import (
    MemoizedProvider,
    SweepResult,
    apply_params,
    expand_grid,
    rank_by_stability,
    rank_key,
    run_sweep,
    stability_scores,
)
from trading_lab.research.weighting import WeightingRule, adaptive_weights
from trading_lab.research.walkforward import (
    WalkForwardFold,
    WalkForwardResult,
    make_folds,
    walk_forward,
)

__all__ = [
    "PermutationResult",
    "format_permutation",
    "permutation_test",
    "ABResult",
    "ab_test",
    "config_diff",
    "format_ab",
    "RegimeReport",
    "format_regimes",
    "regimes_for_run",
    "CostSensitivity",
    "cost_sensitivity",
    "format_costs",
    "AgentEval",
    "evaluate_agents",
    "evaluate_run",
    "format_agent_eval",
    "Reconciliation",
    "format_reconciliation",
    "reconcile",
    "Robustness",
    "bootstrap",
    "format_robustness",
    "robustness_for_run",
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
    "rank_by_stability",
    "stability_scores",
    "run_sweep",
    "walk_forward",
]
