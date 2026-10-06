"""Agent vote weights from their track record (no look-ahead).

``adaptive_weights`` turns each AI agent's directional correctness (see
``research.attribution``) into a new ensemble weight:

    multiplier = 1 + sensitivity x (correctness - 0.5)
    weight     = clamp(current weight x multiplier, min_weight, max_weight)

With the defaults (sensitivity 10), 55% correct gives 1.5x, 45% gives 0.5x
and 40% or worse switches the agent off; no agent can exceed ``max_weight``.
Agents with fewer than ``min_votes`` measurable votes keep their weight: no
evidence, no change. Deterministic strategies are never re-weighted.

In walk-forward (``walk_forward(..., adapt_agent_weights=True)``) the weights
for each test window come only from the training window before it, and the
attribution there only measures votes whose outcome is known inside the
training window. So the weights never see the data they are tested on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from trading_lab.research.attribution import Attribution


@dataclass(frozen=True, slots=True)
class WeightingRule:
    horizon: int = 4  # bars ahead used to judge a vote
    min_votes: int = 30  # measurable votes needed before a weight changes
    sensitivity: float = 10.0
    min_weight: float = 0.0
    max_weight: float = 2.0

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.min_votes < 1 or self.sensitivity < 0:
            raise ValueError("horizon and min_votes must be >= 1, sensitivity >= 0")
        if not 0 <= self.min_weight <= self.max_weight:
            raise ValueError("need 0 <= min_weight <= max_weight")


def adaptive_weights(
    attributions: Mapping[str, Attribution],
    current: Mapping[str, float],
    rule: WeightingRule = WeightingRule(),
) -> dict[str, float]:
    """New weights for the AI agents in ``current`` (deterministic strategies are left out)."""
    out: dict[str, float] = {}
    for name, weight in current.items():
        a = attributions.get(name)
        if a is None or not a.is_agent:
            continue
        if a.measured < rule.min_votes or a.directional_correctness is None:
            out[name] = weight
            continue
        multiplier = max(0.0, 1.0 + rule.sensitivity * (a.directional_correctness - 0.5))
        out[name] = round(min(max(weight * multiplier, rule.min_weight), rule.max_weight), 6)
    non_agents = sum(w for n, w in current.items() if n not in out)
    if non_agents + sum(out.values()) <= 0:  # never leave the ensemble without a voter
        return {n: current[n] for n in out}
    return out
