"""Portfolio-level circuit breakers.

They only ever block **new entries**. Exits, including stop-losses, always
stay possible.

* **Max drawdown kill switch:** once equity at a bar close is
  ``max_drawdown_pct`` below its running peak, no new entries are allowed for
  the rest of the run. With ``flatten_on_halt`` the session also schedules
  exits for every open position.
* **Daily loss limit:** once equity at a bar close is ``daily_loss_limit_pct``
  below the equity at the start of that UTC day, new entries are blocked until
  the next UTC day. A day starts at 00:00 UTC, and its starting equity is the
  equity at the last close before that moment.
* **Stop-loss cooldown:** after a stop-out on bar ``t``, that symbol accepts no
  new entries at the opens of the next ``stop_loss_cooldown_bars`` bars.

Breakers are checked against bar-close equity, the same marks used for the
equity curve. The state is plain JSON so live runs can persist and resume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from trading_lab.config import RiskConfig


@dataclass(slots=True)
class BreakerState:
    peak_equity: float
    day: date | None = None
    day_start_equity: float = 0.0
    last_equity: float = 0.0
    halted_reason: str | None = None
    daily_blocked_day: date | None = None
    cooldown_until: dict[str, datetime] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "peak_equity": self.peak_equity,
            "day": self.day.isoformat() if self.day else None,
            "day_start_equity": self.day_start_equity,
            "last_equity": self.last_equity,
            "halted_reason": self.halted_reason,
            "daily_blocked_day": self.daily_blocked_day.isoformat() if self.daily_blocked_day else None,
            "cooldown_until": {s: t.isoformat() for s, t in self.cooldown_until.items()},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BreakerState:
        return cls(
            peak_equity=data["peak_equity"],
            day=date.fromisoformat(data["day"]) if data.get("day") else None,
            day_start_equity=data.get("day_start_equity", 0.0),
            last_equity=data.get("last_equity", 0.0),
            halted_reason=data.get("halted_reason"),
            daily_blocked_day=(
                date.fromisoformat(data["daily_blocked_day"]) if data.get("daily_blocked_day") else None
            ),
            cooldown_until={
                s: datetime.fromisoformat(t) for s, t in data.get("cooldown_until", {}).items()
            },
        )


class CircuitBreakers:
    def __init__(
        self,
        config: RiskConfig,
        bar_length: timedelta,
        initial_equity: float,
        state: BreakerState | None = None,
    ) -> None:
        self._config = config
        self._bar = bar_length
        self.state = state or BreakerState(
            peak_equity=initial_equity, day_start_equity=initial_equity, last_equity=initial_equity
        )

    @property
    def halted(self) -> bool:
        return self.state.halted_reason is not None

    def entry_block_reason(self, symbol: str, bar_open: datetime) -> str | None:
        """Why a new entry at the open of ``bar_open`` is not allowed, or None."""
        st = self.state
        if st.halted_reason is not None:
            return f"circuit breaker: {st.halted_reason}"
        if st.daily_blocked_day is not None and bar_open.date() == st.daily_blocked_day:
            return f"circuit breaker: daily loss limit hit on {st.daily_blocked_day}"
        until = st.cooldown_until.get(symbol)
        if until is not None and bar_open < until:
            return f"stop-loss cooldown until {until:%Y-%m-%d %H:%M}"
        return None

    def on_stop_loss(self, symbol: str, bar_open: datetime) -> None:
        bars = self._config.stop_loss_cooldown_bars
        if bars > 0:
            self.state.cooldown_until[symbol] = bar_open + (bars + 1) * self._bar

    def on_bar_close(self, bar_open: datetime, equity: float) -> list[str]:
        """Update with bar-close equity. Returns the breakers that tripped on this bar."""
        cfg, st = self._config, self.state
        close_time = bar_open + self._bar
        today = close_time.date() if close_time.time() != datetime.min.time() else (
            close_time - timedelta(microseconds=1)
        ).date()
        if st.day != today:  # first close of a new UTC day
            st.day = today
            st.day_start_equity = st.last_equity
        st.last_equity = equity
        st.peak_equity = max(st.peak_equity, equity)

        events = []
        if cfg.max_drawdown_pct > 0 and st.halted_reason is None:
            drawdown = 1.0 - equity / st.peak_equity
            if drawdown >= cfg.max_drawdown_pct:
                st.halted_reason = (
                    f"max drawdown {drawdown:.1%} >= {cfg.max_drawdown_pct:.0%} "
                    f"(peak {st.peak_equity:,.2f}, equity {equity:,.2f})"
                )
                events.append(st.halted_reason)
        if cfg.daily_loss_limit_pct > 0 and st.daily_blocked_day != today and st.day_start_equity > 0:
            daily_loss = 1.0 - equity / st.day_start_equity
            if daily_loss >= cfg.daily_loss_limit_pct:
                st.daily_blocked_day = today
                events.append(
                    f"daily loss {daily_loss:.1%} >= {cfg.daily_loss_limit_pct:.0%} on {today}; "
                    "new entries paused until the next UTC day"
                )
        return events
