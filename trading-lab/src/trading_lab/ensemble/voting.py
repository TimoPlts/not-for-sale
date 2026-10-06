"""Voting engine: combines per-strategy signals into one ensemble signal.

Scoring, for signals ``i`` with weight ``w_i`` and confidence ``c_i``:

    W          = sum of w_i over all participating signals (HOLD votes count too)
    buy_score  = sum(w_i * c_i for BUY signals)  / W
    sell_score = sum(w_i * c_i for SELL signals) / W
    net        = buy_score - sell_score

* BUY  if ``net >= buy_threshold`` and at least ``min_agreeing`` strategies say BUY
* SELL if ``-net >= sell_threshold`` and at least ``min_agreeing`` strategies say SELL
* HOLD otherwise

Abstentions (HOLD) dilute the score, and opposing votes cancel out. The
ensemble's confidence is ``|net|``. The full vote breakdown goes into the
metadata so every decision can be audited. Any producer of ``Signal``s, such
as a future AI agent, can take part by being given a weight.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from trading_lab.config import StrategySpec, VotingConfig
from trading_lab.core.models import Direction, Signal

ENSEMBLE_NAME = "ensemble"


class VotingEngine:
    def __init__(self, weights: Mapping[str, float], config: VotingConfig | None = None) -> None:
        if not weights:
            raise ValueError("voting engine needs at least one weighted strategy")
        for name, weight in weights.items():
            if not (isinstance(weight, (int, float)) and weight >= 0):
                raise ValueError(f"weight for {name!r} must be >= 0, got {weight!r}")
        if sum(weights.values()) <= 0:
            raise ValueError("total strategy weight must be positive")
        self._weights = dict(weights)
        self._config = config or VotingConfig()

    @classmethod
    def from_specs(cls, specs: Iterable[StrategySpec], config: VotingConfig) -> VotingEngine:
        return cls({s.name: s.weight for s in specs if s.enabled and s.weight > 0}, config)

    @property
    def weights(self) -> Mapping[str, float]:
        return dict(self._weights)

    def combine(self, signals: Sequence[Signal]) -> Signal:
        if not signals:
            raise ValueError("cannot combine an empty list of signals")
        symbol, timestamp = signals[0].symbol, signals[0].timestamp
        seen: set[str] = set()
        for sig in signals:
            if sig.symbol != symbol or sig.timestamp != timestamp:
                raise ValueError("all signals must share the same symbol and timestamp")
            if sig.strategy not in self._weights:
                raise ValueError(f"no voting weight configured for strategy {sig.strategy!r}")
            if sig.strategy in seen:
                raise ValueError(f"duplicate signal from strategy {sig.strategy!r}")
            seen.add(sig.strategy)

        total_weight = sum(self._weights[s.strategy] for s in signals)
        if total_weight <= 0:
            raise ValueError("participating strategies have zero total weight")

        buy_score = sum(
            self._weights[s.strategy] * s.confidence for s in signals if s.direction is Direction.BUY
        ) / total_weight
        sell_score = sum(
            self._weights[s.strategy] * s.confidence
            for s in signals
            if s.direction is Direction.SELL
        ) / total_weight
        net = buy_score - sell_score
        n_buy = sum(s.direction is Direction.BUY for s in signals)
        n_sell = sum(s.direction is Direction.SELL for s in signals)

        cfg = self._config
        if net >= cfg.buy_threshold and n_buy >= cfg.min_agreeing:
            direction = Direction.BUY
        elif -net >= cfg.sell_threshold and n_sell >= cfg.min_agreeing:
            direction = Direction.SELL
        else:
            direction = Direction.HOLD

        metadata: dict[str, Any] = {
            "net_score": net,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "buy_votes": n_buy,
            "sell_votes": n_sell,
            "hold_votes": len(signals) - n_buy - n_sell,
            "votes": [
                {
                    "strategy": s.strategy,
                    "direction": s.direction.value,
                    "confidence": s.confidence,
                    "weight": self._weights[s.strategy],
                }
                for s in signals
            ],
        }
        confidence = min(abs(net), 1.0) if direction is not Direction.HOLD else 0.0
        return Signal(ENSEMBLE_NAME, symbol, direction, confidence, timestamp, metadata)
