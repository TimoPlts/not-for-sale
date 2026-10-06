"""Model usage accounting: calls, cache hits, failures, retries, latency and tokens.

Two sources, one format (``UsageStats`` per agent):

* ``UsageTracker``: lives on a provider (``provider.usage``) and counts
  everything that goes through it during a command, e.g. a whole sweep.
* ``usage_from_signals``: rebuilds the same numbers for a stored run from
  its signal metadata (``cache`` and ``llm`` keys), so reports work later.

Token counts come from the endpoint when it reports them (``usage`` in an
OpenAI-compatible response). Otherwise they are estimated as
``ceil(characters / 3)``. Real ratios are closer to 3.5–4 characters per
token for English and JSON, so the estimate errs on the high side.
Estimated numbers are always marked as such.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable, Mapping

CHARS_PER_TOKEN_ESTIMATE = 3.0


def estimate_tokens(chars: int) -> int:
    return int(math.ceil(max(chars, 0) / CHARS_PER_TOKEN_ESTIMATE))


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One model call made by an agent (stored in the signal metadata as ``llm``)."""

    provider: str
    model: str
    ok: bool  # the endpoint answered (the answer may still be invalid)
    latency_seconds: float
    attempts: int
    input_chars: int
    output_chars: int
    input_tokens: int
    output_tokens: int
    tokens_estimated: bool
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "latency_seconds": round(self.latency_seconds, 4)}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> CallRecord:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass(slots=True)
class UsageStats:
    calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    failures: int = 0  # the endpoint did not answer (after retries)
    invalid_answers: int = 0  # answered, but not valid structured JSON
    retries: int = 0
    input_chars: int = 0
    output_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_calls: int = 0  # calls whose token counts are estimates
    total_latency_seconds: float = 0.0

    def add_call(self, record: CallRecord, *, valid: bool) -> None:
        self.calls += 1
        self.retries += max(record.attempts - 1, 0)
        self.total_latency_seconds += record.latency_seconds
        self.input_chars += record.input_chars
        self.output_chars += record.output_chars
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.estimated_calls += int(record.tokens_estimated)
        if not record.ok:
            self.failures += 1
        elif not valid:
            self.invalid_answers += 1

    def add_cache(self, hit: bool) -> None:
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def merge(self, other: UsageStats) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    @property
    def avg_latency_seconds(self) -> float | None:
        return self.total_latency_seconds / self.calls if self.calls else None

    @property
    def tokens_estimated(self) -> bool:
        return self.estimated_calls > 0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "avg_latency_seconds": self.avg_latency_seconds,
                "tokens_estimated": self.tokens_estimated}


def total_usage(per_agent: Mapping[str, UsageStats]) -> UsageStats:
    total = UsageStats()
    for stats in per_agent.values():
        total.merge(stats)
    return total


class UsageTracker:
    """Per-agent usage of one provider during this process."""

    def __init__(self) -> None:
        self.per_agent: dict[str, UsageStats] = {}

    def agent(self, name: str) -> UsageStats:
        return self.per_agent.setdefault(name, UsageStats())

    def record_call(self, agent: str, record: CallRecord, *, valid: bool) -> None:
        self.agent(agent).add_call(record, valid=valid)

    def record_cache(self, agent: str, hit: bool) -> None:
        self.agent(agent).add_cache(hit)

    def total(self) -> UsageStats:
        return total_usage(self.per_agent)


def usage_from_signals(signals: Iterable[tuple[str, Mapping[str, Any]]]) -> dict[str, UsageStats]:
    """Usage per agent from ``(strategy, metadata)`` pairs of a run's signals."""
    out: dict[str, UsageStats] = {}
    for strategy, meta in signals:
        cache = meta.get("cache")
        llm = meta.get("llm")
        if cache is None and llm is None:
            continue
        stats = out.setdefault(strategy, UsageStats())
        if cache == "hit":
            stats.add_cache(True)
        elif cache in ("stored", "miss"):
            stats.add_cache(False)
        if isinstance(llm, Mapping):
            stats.add_call(CallRecord.from_json(llm), valid="error" not in meta)
    return out


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def format_usage(per_agent: Mapping[str, UsageStats], title: str = "Model usage") -> str:
    total = total_usage(per_agent)
    est = " (estimated)" if total.tokens_estimated else ""
    avg = "n/a" if total.avg_latency_seconds is None else f"{total.avg_latency_seconds:.2f}s"
    lines = [
        f"{title}:",
        f"  calls: {total.calls}   cache hits: {total.cache_hits}   cache misses: {total.cache_misses}   "
        f"failures: {total.failures}   invalid answers: {total.invalid_answers}   retries: {total.retries}",
        f"  avg latency: {avg}   total latency: {_duration(total.total_latency_seconds)}",
        f"  input tokens: {total.input_tokens:,}{est}   output tokens: {total.output_tokens:,}{est}   "
        f"(input chars {total.input_chars:,}, output chars {total.output_chars:,})",
    ]
    if len(per_agent) > 1:
        lines.append(f"  {'agent':<16} {'calls':>6} {'hits':>6} {'misses':>6} {'fail':>5} {'retry':>5} "
                     f"{'avg lat':>8} {'in tok':>9} {'out tok':>8}")
        for name, s in sorted(per_agent.items()):
            lat = "n/a" if s.avg_latency_seconds is None else f"{s.avg_latency_seconds:.2f}s"
            lines.append(f"  {name:<16} {s.calls:>6} {s.cache_hits:>6} {s.cache_misses:>6} {s.failures:>5} "
                         f"{s.retries:>5} {lat:>8} {s.input_tokens:>9,} {s.output_tokens:>8,}")
    return "\n".join(lines)
