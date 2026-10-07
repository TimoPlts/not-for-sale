"""Stage 21C: config presets (trading-lab init-config)."""

import tomllib
from datetime import datetime, timedelta, timezone

import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig, load_config
from trading_lab.data import SyntheticProvider
from trading_lab.presets import PRESETS, _toml_value, preset_toml, render
from trading_lab.research import config_diff

UTC = timezone.utc


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_round_trips_and_runs(name, tmp_path):
    path = tmp_path / f"{name}.toml"
    path.write_text(preset_toml(name))
    cfg = load_config(path)
    assert cfg == PRESETS[name].config() and cfg != AppConfig()
    assert path.read_text().startswith(f"# trading-lab preset '{name}'")
    start = datetime(2024, 2, 1, tzinfo=UTC)
    small = cfg.with_overrides({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})
    assert BacktestEngine(small, SyntheticProvider(seed=3)).run(start, start + timedelta(days=12)).metrics


def test_files_list_only_the_changes():
    data = tomllib.loads(preset_toml("conservative"))
    assert set(data) == {"risk"}  # strategies, market, ... keep their defaults
    assert data["risk"] == {"risk_per_trade_pct": 0.005, "max_position_pct": 0.15, "position_volatility_pct": 0.1,
                            "stop_mode": "atr", "trend_filter_period": 200, "max_drawdown_pct": 0.15,
                            "daily_loss_limit_pct": 0.03}
    trend = tomllib.loads(preset_toml("trend"))
    assert {"rsi", "macd", "bollinger", "donchian", "ma_cross"} <= set(trend["strategies"])  # written in full
    assert {k for k, _, _ in config_diff(AppConfig(), PRESETS["trend-shorts"].config())} >= {"risk.allow_short"}
    mr = tomllib.loads(preset_toml("mean-reversion"))
    assert mr["voting"] == {"regime_weights": {"down": {"bollinger": 0.25, "rsi": 0.25}}}
    assert render(AppConfig()) == "\n"  # nothing differs
    with pytest.raises(TypeError):
        _toml_value(None)


def test_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init-config"]) == 0
    listing = capsys.readouterr().out
    assert all(name in listing for name in PRESETS)
    assert main(["init-config", "trend"]) == 0
    out = capsys.readouterr().out
    assert (tmp_path / "config" / "trend.toml").exists() and "checkup" in out and "trading-lab ab" in out
    assert main(["init-config", "trend"]) == 1  # no silent overwrite
    (tmp_path / "config" / "trend.toml").write_text("# mine\n")
    assert main(["init-config", "trend", "--force"]) == 0
    assert load_config(tmp_path / "config" / "trend.toml") == PRESETS["trend"].config()
    assert main(["init-config", "conservative", "elsewhere/c.toml"]) == 0
    assert (tmp_path / "elsewhere" / "c.toml").exists()
    assert main(["init-config", "yolo"]) == 1
