"""Ready-made configs to start experiments from (``trading-lab init-config``).

Each preset is a set of overrides on top of the built-in defaults. It is
written as a short TOML file that lists only what differs, so you can see at
a glance what the preset changes. Everything else keeps the defaults
documented in ``config/default.toml``. All presets are paper trading, like
everything here. They are starting points to test with ``checkup`` and
``ab``, not recommendations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from trading_lab.config import AppConfig


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    summary: str
    overrides: Mapping[str, Mapping[str, Any]]

    def config(self, base: AppConfig | None = None) -> AppConfig:
        return (base or AppConfig()).with_overrides(self.overrides)


PRESETS: dict[str, Preset] = {p.name: p for p in (
    Preset("trend", "trend following: Donchian breakouts and an EMA crossover join the vote, with a trailing "
           "stop and a time stop", {
               "strategies": {"donchian": {"weight": 1.5, "entry_period": 20, "exit_period": 10},
                              "ma_cross": {"weight": 1.0, "fast": 20, "slow": 50, "average": "ema"}},
               "risk": {"trailing_stop_pct": 0.04, "trailing_activation_pct": 0.02, "max_holding_bars": 72},
           }),
    Preset("trend-shorts", "the trend preset trading both directions (simulated shorts, fully collateralised)", {
        "strategies": {"donchian": {"weight": 1.5, "entry_period": 20, "exit_period": 10},
                       "ma_cross": {"weight": 1.0, "fast": 20, "slow": 50, "average": "ema"}},
        "risk": {"allow_short": True, "trailing_stop_pct": 0.04, "trailing_activation_pct": 0.02,
                 "max_holding_bars": 72},
    }),
    Preset("conservative", "smaller, volatility-targeted positions, ATR stops, a 200-bar trend filter and "
           "tighter circuit breakers", {
               "risk": {"risk_per_trade_pct": 0.005, "max_position_pct": 0.15, "position_volatility_pct": 0.10,
                        "stop_mode": "atr", "trend_filter_period": 200, "max_drawdown_pct": 0.15,
                        "daily_loss_limit_pct": 0.03},
           }),
    Preset("mean-reversion", "RSI and Bollinger only, quick profits and a one-day time stop; mean reversion "
           "is muted in clear down-trends", {
               "strategies": {"macd": {"enabled": False}},
               "risk": {"take_profit_pct": 0.03, "max_holding_bars": 24},
               "voting": {"regime_weights": {"down": {"rsi": 0.25, "bollinger": 0.25}}},
           }),
)}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a valid TOML basic string
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {value!r} to TOML")


def _table(name: str, values: Mapping[str, Any]) -> list[str]:
    scalars = [f"{k} = {_toml_value(v)}" for k, v in values.items() if not isinstance(v, Mapping) and v is not None]
    lines = [f"[{name}]", *scalars] if scalars or not values else []  # sub-tables need no empty parent header
    for k, v in values.items():
        if isinstance(v, Mapping):
            lines += ["", *_table(f"{name}.{k}", v)] if lines else _table(f"{name}.{k}", v)
    return lines


def render(config: AppConfig, *, header: str = "", base: AppConfig | None = None) -> str:
    """TOML with only the settings that differ from ``base`` (default: the built-in defaults).

    The strategies table is written in full when it differs, because a
    ``[strategies]`` table replaces the default strategies rather than adding to them.
    """
    base_map, new_map = (base or AppConfig()).to_mapping(), config.to_mapping()
    out = [f"# {line}" if line else "#" for line in header.splitlines()]
    for section, values in new_map.items():
        if section == "strategies":
            continue
        diff = {k: v for k, v in values.items() if base_map.get(section, {}).get(k) != v}
        if diff:
            out += ["", *_table(section, diff)]
    if new_map["strategies"] != base_map["strategies"]:
        for name, table in new_map["strategies"].items():
            out += ["", *_table(f"strategies.{name}", table)]
    return "\n".join(out).lstrip("\n") + "\n"


def preset_toml(name: str) -> str:
    preset = PRESETS[name]
    header = (f"trading-lab preset '{name}': {preset.summary}.\n"
              "Only the settings that differ from the built-in defaults are listed; everything else\n"
              "keeps the defaults documented in config/default.toml. Paper trading only.\n"
              "Test it before using it: trading-lab checkup / trading-lab ab (see README).")
    return render(preset.config(), header=header)
