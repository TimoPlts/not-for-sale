import pytest

from trading_lab.config import AppConfig, load_config
from trading_lab.core.errors import ConfigError
from trading_lab.core.symbols import SUPPORTED_SYMBOLS, timeframe_to_seconds


def test_defaults_match_v1_requirements():
    cfg = load_config()
    assert cfg.portfolio.initial_cash == 10_000.0
    assert cfg.market.symbols == ("BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT")
    assert cfg.market.timeframe == "1h"
    assert cfg.execution.fee_rate == 0.001
    assert cfg.execution.slippage_bps == 5.0


def test_shipped_default_toml_equals_code_defaults(default_config_path):
    assert load_config(default_config_path) == AppConfig()


def test_toml_overrides_and_int_values_become_float(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text(
        '[portfolio]\ninitial_cash = 5000\n'
        '[market]\nsymbols = ["BTC/USDT"]\ntimeframe = "15m"\n'
        "[execution]\nfee_rate = 0.00075\n"
    )
    cfg = load_config(path)
    assert cfg.portfolio.initial_cash == 5000.0
    assert isinstance(cfg.portfolio.initial_cash, float)
    assert cfg.market.symbols == ("BTC/USDT",)
    assert cfg.market.timeframe == "15m"
    assert cfg.execution.fee_rate == 0.00075
    assert cfg.risk == AppConfig().risk  # untouched sections keep defaults


@pytest.mark.parametrize(
    "data, match",
    [
        ({"market": {"symbols": ["XRP/USDT"]}}, "unsupported symbols"),
        ({"market": {"symbols": ["BTC/USDT", "BTC/USDT"]}}, "duplicates"),
        ({"market": {"timeframe": "7m"}}, "unsupported timeframe"),
        ({"execution": {"fee_rate": -0.001}}, "fee_rate"),
        ({"execution": {"slippage_bps": float("nan")}}, "slippage_bps"),
        ({"risk": {"max_position_pct": 1.5}}, "max_position_pct"),
        ({"risk": {"max_open_positions": 0}}, "max_open_positions"),
        ({"portfolio": {"initial_cash": 0}}, "initial_cash"),
        ({"exchange": {"apiKey": "x"}}, "unknown config section"),
        ({"market": {"api_key": "x"}}, "unknown key"),
    ],
)
def test_invalid_config_is_rejected(data, match):
    with pytest.raises(ConfigError, match=match):
        AppConfig.from_mapping(data)


def test_fingerprint_is_stable_and_sensitive():
    a = AppConfig()
    b = AppConfig.from_mapping({})
    c = AppConfig.from_mapping({"execution": {"fee_rate": 0.002}})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert len(a.fingerprint()) == 64


def test_supported_symbols_and_timeframes():
    assert set(SUPPORTED_SYMBOLS) == {"BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"}
    assert timeframe_to_seconds("1h") == 3600
    assert timeframe_to_seconds("1d") == 86400
    with pytest.raises(ValueError):
        timeframe_to_seconds("2w")


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_default_strategies_and_voting():
    cfg = AppConfig()
    assert [s.name for s in cfg.enabled_strategies] == ["rsi", "macd", "bollinger"]
    assert cfg.strategies[0].params == {"period": 14, "oversold": 30.0, "overbought": 70.0}
    assert cfg.voting.buy_threshold == 0.15 and cfg.voting.min_agreeing == 1


def test_strategies_section_replaces_defaults(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text(
        "[strategies.rsi]\nweight = 2\nperiod = 7\n"
        "[strategies.macd]\nenabled = false\n"
    )
    cfg = load_config(path)
    rsi, macd = cfg.strategies
    assert rsi.weight == 2.0 and rsi.params == {"period": 7}
    assert not macd.enabled
    assert [s.name for s in cfg.enabled_strategies] == ["rsi"]


@pytest.mark.parametrize(
    "data, match",
    [
        ({"strategies": {"rsi": {"enabled": False}}}, "at least one strategy"),
        ({"strategies": {"rsi": {"weight": -1}}}, "weight"),
        ({"strategies": {"rsi": {"weight": 0}}}, "positive total weight"),
        ({"strategies": {"rsi": 5}}, "must be a table"),
        ({"strategies": {"bad-name": {}}}, "identifier"),
        ({"voting": {"buy_threshold": 0}}, "buy_threshold"),
        ({"voting": {"min_agreeing": 0}}, "min_agreeing"),
        ({"data": {"page_limit": 0}}, "page_limit"),
        ({"data": {"use_cache": "yes"}}, "use_cache"),
    ],
)
def test_invalid_stage2_config_is_rejected(data, match):
    with pytest.raises(ConfigError, match=match):
        AppConfig.from_mapping(data)


def test_mapping_round_trip_and_overrides():
    cfg = AppConfig()
    assert AppConfig.from_mapping(cfg.to_mapping()) == cfg
    changed = cfg.with_overrides({"market": {"timeframe": "15m"}, "risk": {"stop_loss_pct": 0.03}})
    assert changed.market.timeframe == "15m" and changed.risk.stop_loss_pct == 0.03
    assert changed.strategies == cfg.strategies and cfg.market.timeframe == "1h"
    with pytest.raises(ConfigError):
        cfg.with_overrides({"market": {"timeframe": "7m"}})


def test_from_dict_round_trip():
    cfg = AppConfig().with_overrides({"strategies": {"rsi": {"weight": 2.0, "period": 7}}})
    assert AppConfig.from_dict(cfg.to_dict()) == cfg
    assert AppConfig.from_dict(cfg.to_dict()).fingerprint() == cfg.fingerprint()
