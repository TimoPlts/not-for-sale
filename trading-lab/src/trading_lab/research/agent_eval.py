"""Answer-quality diagnostics for the AI agents of a stored run.

``agent-report`` asks whether an agent's votes were *right*. This module
asks whether its answers are *sound*, whatever the market did afterwards:

* **availability:** decisions without a usable answer, by error type;
* **consistency:** votes that contradict the agent's own label (e.g. BUY
  with ``regime = bearish_trend``), and BUY/SELL votes on a label for which
  the prompt asks for HOLD;
* **spread:** one-sided voting (almost only BUY, only SELL or only HOLD),
  confidence that barely varies, BUY/SELL votes with confidence 0 (which
  carry no weight);
* **explanations:** empty or very short rationales, and one rationale
  repeated for most answers (boilerplate);
* **model calls:** latency and failed calls, when the run recorded them.

The label rules come from the agent classes (``contradicting_votes`` and
``hold_labels``). Agents without such rules get every check except
consistency. Distribution checks need at least ``min_answers`` answers, so
a short run is not judged on a handful of votes. Everything is read-only.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import trading_lab.agents  # noqa: F401  (registers the agent classes looked up below)
from trading_lab.strategies.registry import strategy_class

SHORT_RATIONALE = 15  # characters
_ERROR_KIND = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*): ")


def error_kind(message: str) -> str:
    """``TransportTimeout: ...`` -> ``TransportTimeout``; other messages are kept (shortened)."""
    match = _ERROR_KIND.match(message)
    return match.group(1) if match else message[:60]


@dataclass
class AgentEval:
    strategy: str
    label: str | None = None
    decisions: int = 0  # decision bars (answered or not)
    errors: Counter = field(default_factory=Counter)  # error kind -> count
    directions: Counter = field(default_factory=Counter)  # of the answers
    labels: Counter = field(default_factory=Counter)
    missing_label: int = 0
    confidences: list[float] = field(default_factory=list)  # of the BUY/SELL answers
    zero_confidence_votes: int = 0
    short_rationales: int = 0
    top_rationale: str = ""
    top_rationale_count: int = 0
    contradictions: list[str] = field(default_factory=list)
    hold_label_votes: int = 0
    calls: int = 0
    failed_calls: int = 0
    latency_total: float = 0.0
    min_answers: int = 20

    @property
    def answered(self) -> int:
        return sum(self.directions.values())

    @property
    def directional(self) -> int:
        return self.directions["BUY"] + self.directions["SELL"]

    @property
    def mean_latency(self) -> float | None:
        return self.latency_total / self.calls if self.calls else None

    @property
    def warnings(self) -> list[str]:
        out = []
        unanswered = sum(self.errors.values())
        if self.decisions and unanswered / self.decisions > 0.1:
            kind, n = self.errors.most_common(1)[0]
            out.append(f"{unanswered / self.decisions:.0%} of decisions had no usable answer (mostly {kind}: {n})")
        if self.contradictions:
            out.append(f"{len(self.contradictions)} vote(s) contradict the agent's own {self.label}")
        if self.directional and self.hold_label_votes / self.directional > 0.25:
            out.append(f"{self.hold_label_votes} BUY/SELL vote(s) on a {self.label} that calls for HOLD")
        if self.missing_label:
            out.append(f"{self.missing_label} answer(s) without a {self.label}")
        if self.zero_confidence_votes:
            out.append(f"{self.zero_confidence_votes} BUY/SELL vote(s) with confidence 0 (no weight)")
        if self.short_rationales:
            out.append(f"{self.short_rationales} empty or very short rationale(s)")
        if self.answered >= self.min_answers:
            hold = self.directions["HOLD"] / self.answered
            if hold >= 0.95:
                out.append(f"almost always HOLD ({hold:.0%}): the agent barely takes part")
            if self.top_rationale_count / self.answered > 0.5:
                out.append(f"the same rationale in {self.top_rationale_count / self.answered:.0%} of answers")
        if self.directional >= self.min_answers:
            side, n = max((("BUY", self.directions["BUY"]), ("SELL", self.directions["SELL"])), key=lambda x: x[1])
            if n / self.directional >= 0.9:
                out.append(f"one-sided: {n / self.directional:.0%} of its BUY/SELL votes are {side}")
            distinct = sorted(set(round(c, 3) for c in self.confidences))
            if len(distinct) <= 2:
                out.append("confidence barely varies (" + ", ".join(f"{c:g}" for c in distinct) + ")")
        return out

    def to_dict(self) -> dict[str, Any]:
        conf = pd.Series(self.confidences, dtype="float64")
        return {
            "strategy": self.strategy, "label": self.label, "decisions": self.decisions,
            "answered": self.answered, "errors": dict(self.errors), "directions": dict(self.directions),
            "labels": dict(self.labels), "confidence_mean": None if conf.empty else float(conf.mean()),
            "confidence_min": None if conf.empty else float(conf.min()),
            "confidence_max": None if conf.empty else float(conf.max()),
            "contradictions": list(self.contradictions), "hold_label_votes": self.hold_label_votes,
            "short_rationales": self.short_rationales, "top_rationale_count": self.top_rationale_count,
            "calls": self.calls, "failed_calls": self.failed_calls, "mean_latency": self.mean_latency,
            "warnings": self.warnings,
        }


def evaluate_agents(signals: pd.DataFrame, *, min_answers: int = 20) -> dict[str, AgentEval]:
    """Diagnostics per agent from a ``load_signals`` frame (other strategies are ignored)."""
    out: dict[str, AgentEval] = {}
    rationales: dict[str, Counter] = {}
    ordered = signals.sort_values(["timestamp", "symbol"], kind="stable") if not signals.empty else signals
    for row in ordered.itertuples(index=False):
        meta = json.loads(row.metadata_json)
        if "agent" not in (meta.get("params") or {}) or "cache" not in meta:
            continue  # not an agent, or not a decision bar
        name = str(row.strategy)
        cls = strategy_class(name)
        ev = out.get(name)
        if ev is None:
            ev = out[name] = AgentEval(name, label=getattr(cls, "label", None), min_answers=min_answers)
            rationales[name] = Counter()
        ev.decisions += 1
        llm = meta.get("llm")
        if isinstance(llm, dict):
            ev.calls += 1
            ev.failed_calls += 0 if llm.get("ok") else 1
            ev.latency_total += float(llm.get("latency_seconds") or 0.0)
        if "error" in meta:
            ev.errors[error_kind(str(meta["error"]))] += 1
            continue
        direction = str(row.direction).upper()
        ev.directions[direction] += 1
        rationale = str(meta.get("rationale") or "").strip()
        if len(rationale) < SHORT_RATIONALE:
            ev.short_rationales += 1
        rationales[name][rationale] += 1
        if direction != "HOLD":
            ev.confidences.append(float(row.confidence))
            if float(row.confidence) == 0:
                ev.zero_confidence_votes += 1
        if ev.label is None:
            continue
        value = meta.get(ev.label)
        if value is None:
            ev.missing_label += 1
            continue
        ev.labels[str(value)] += 1
        if getattr(cls, "contradicting_votes", {}).get(str(value)) == direction:
            ts = pd.Timestamp(row.timestamp)
            ev.contradictions.append(f"{ts:%Y-%m-%d %H:%M} {row.symbol} {direction} with {ev.label}={value}")
        elif direction != "HOLD" and str(value) in getattr(cls, "hold_labels", ()):
            ev.hold_label_votes += 1
    for name, ev in out.items():
        if rationales[name]:
            ev.top_rationale, ev.top_rationale_count = rationales[name].most_common(1)[0]
    return dict(sorted(out.items()))


def evaluate_run(store: Any, run_id: str, *, min_answers: int = 20) -> dict[str, AgentEval]:
    if store.get_run(run_id) is None:
        raise ValueError(f"unknown run id {run_id!r}")
    return evaluate_agents(store.load_signals(run_id), min_answers=min_answers)


def format_agent_eval(ev: AgentEval, limit: int = 5) -> str:
    lines = [f"Agent: {ev.strategy}"]
    unanswered = sum(ev.errors.values())
    lines.append(f"  decisions: {ev.decisions}, answered {ev.answered}"
                 + (f", no usable answer {unanswered} (" + ", ".join(f"{k}: {n}" for k, n in
                                                            ev.errors.most_common()) + ")" if unanswered else ""))
    if ev.answered:
        lines.append("  votes: " + ", ".join(f"{d} {ev.directions[d]} ({ev.directions[d] / ev.answered:.0%})"
                                             for d in ("BUY", "SELL", "HOLD")))
    if ev.confidences:
        c = pd.Series(ev.confidences)
        lines.append(f"  BUY/SELL confidence: mean {c.mean():.2f}, range {c.min():.2f}-{c.max():.2f}, "
                     f"{c.round(3).nunique()} distinct value(s)")
    if ev.labels:
        lines.append(f"  {ev.label}: " + ", ".join(f"{k} {n}" for k, n in ev.labels.most_common()))
    if ev.calls:
        lines.append(f"  model calls: {ev.calls}, failed {ev.failed_calls}, mean latency {ev.mean_latency:.2f}s")
    if ev.top_rationale_count > 1:
        lines.append(f"  most repeated rationale ({ev.top_rationale_count}x): {ev.top_rationale[:80]!r}")
    for item in ev.contradictions[:limit]:
        lines.append(f"    contradiction: {item}")
    if len(ev.contradictions) > limit:
        lines.append(f"    ... and {len(ev.contradictions) - limit} more")
    warnings = ev.warnings
    lines += [f"  ! {w}" for w in warnings] if warnings else ["  no problems found"]
    return "\n".join(lines)
