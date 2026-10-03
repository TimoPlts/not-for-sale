"""Show current strategy signals for all configured pairs, from public market data.

Usage:
    python scripts/live_signals.py [--timeframe 1h] [--days 30] [--config config/default.toml]

Read-only: fetches public candles via CCXT (no API keys) and prints signals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from trading_lab.config import AppConfig, load_config
from trading_lab.data import build_provider
from trading_lab.ensemble import VotingEngine
from trading_lab.strategies import build_strategies

INDICATOR_KEYS = {"rsi": ["rsi"], "macd": ["hist", "crossover"], "bollinger": ["percent_b"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--timeframe", help="override the configured timeframe, e.g. 15m, 4h")
    parser.add_argument("--days", type=int, default=30, help="history to load (default 30)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.timeframe:
        data = cfg.to_dict()
        data["market"]["timeframe"] = args.timeframe
        data["strategies"] = {s.name: {"enabled": s.enabled, "weight": s.weight, **s.params}
                              for s in cfg.strategies}
        cfg = AppConfig.from_mapping(data)

    provider = build_provider(cfg)
    strategies = build_strategies(cfg.enabled_strategies)
    voting = VotingEngine.from_specs(cfg.strategies, cfg.voting)
    since = datetime.now(timezone.utc) - timedelta(days=args.days)

    print(f"exchange={provider.name}  timeframe={cfg.market.timeframe}  history={args.days}d\n")
    for symbol in cfg.market.symbols:
        candles = provider.fetch_ohlcv(symbol, cfg.market.timeframe, since)
        last = candles.iloc[-1]
        print(f"{symbol}  last closed candle {candles.index[-1]:%Y-%m-%d %H:%M} UTC  "
              f"close={last['close']:.6g}  ({len(candles)} candles)")
        signals = [s.generate_signal(symbol, candles) for s in strategies]
        for sig in signals:
            values = {k: sig.metadata.get(k) for k in INDICATOR_KEYS.get(sig.strategy, [])}
            shown = "  ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in values.items()
            )
            print(f"   {sig.strategy:<10} {sig.direction.value.upper():<5} "
                  f"conf={sig.confidence:.2f}   {shown}")
        ens = voting.combine(signals)
        print(f"   {'ENSEMBLE':<10} {ens.direction.value.upper():<5} conf={ens.confidence:.2f}   "
              f"net_score={ens.metadata['net_score']:+.3f}\n")


if __name__ == "__main__":
    main()
