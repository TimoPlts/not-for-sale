"""Walk through one simulated trade at the current public price.

Usage:
    python scripts/paper_trade_demo.py [--symbol BTC/USDT] [--exit-move 0.03]

Shows risk sizing, the entry fill (slippage + fee), the stop price, and the
accounting after exiting at +exit-move and at the stop. Nothing is sent to
any exchange; the portfolio exists only in memory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from trading_lab.config import load_config
from trading_lab.data import build_provider
from trading_lab.execution import CostModel, PaperExecutor
from trading_lab.portfolio import Portfolio
from trading_lab.risk import RiskManager


def run(cfg, symbol: str, entry_price: float, exit_price: float | None, label: str) -> None:
    """Enter at ``entry_price`` and exit at ``exit_price`` (None = exactly at the stop)."""
    costs = CostModel.from_config(cfg.execution)
    portfolio = Portfolio(cfg.portfolio.initial_cash)
    executor = PaperExecutor(portfolio, costs, min_notional=cfg.execution.min_notional)
    risk = RiskManager(cfg.risk, costs, min_notional=cfg.execution.min_notional)
    now = datetime.now(timezone.utc)

    decision = risk.evaluate_entry(symbol, entry_price, portfolio, prices={})
    if not decision.approved:
        print(f"entry rejected: {decision.reason}")
        return
    entry = executor.submit(decision.to_order(now), entry_price).fill
    if exit_price is None:
        exit_price = decision.stop_price
    exit_ = executor.submit(
        risk.evaluate_exit(symbol, portfolio).to_order(now + timedelta(hours=1)), exit_price
    ).fill
    trade = portfolio.closed_trades[-1]

    print(f"--- scenario: {label} ---")
    print(f"  sizing limited by : {decision.sizing['binding_limit']}")
    print(f"  bought            : {entry.quantity:.6f} @ {entry.fill_price:,.4f} "
          f"(ref {entry.reference_price:,.4f}, fee {entry.fee:.4f})")
    print(f"  position value    : {entry.notional:,.2f} USDT "
          f"({entry.notional / cfg.portfolio.initial_cash:.1%} of equity)")
    print(f"  stop price        : {decision.stop_price:,.4f}")
    print(f"  sold              : {exit_.quantity:.6f} @ {exit_.fill_price:,.4f} "
          f"(ref {exit_.reference_price:,.4f}, fee {exit_.fee:.4f})")
    print(f"  realized PnL      : {trade.pnl:+,.2f} USDT ({trade.return_pct:+.2%})")
    print(f"  fees paid         : {portfolio.fees_paid:.2f}   "
          f"slippage cost: {entry.slippage_cost + exit_.slippage_cost:.2f}")
    print(f"  final cash/equity : {portfolio.cash:,.2f} USDT\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--exit-move", type=float, default=0.03, help="winning exit, e.g. 0.03 = +3%%")
    args = parser.parse_args()

    cfg = load_config(args.config)
    provider = build_provider(cfg)
    candles = provider.fetch_ohlcv(
        args.symbol, cfg.market.timeframe, datetime.now(timezone.utc) - timedelta(days=2)
    )
    price = float(candles["close"].iloc[-1])
    print(f"{args.symbol} last close: {price:,.4f}  "
          f"(fee {cfg.execution.fee_rate:.2%}, slippage {cfg.execution.slippage_bps} bps)\n")

    run(cfg, args.symbol, price, price * (1 + args.exit_move), f"price rises {args.exit_move:+.1%}")
    run(cfg, args.symbol, price, None, "stop-loss hit (should lose ~1% of equity)")
    run(cfg, args.symbol, price, price, "flat price (loss = fees + slippage only)")


if __name__ == "__main__":
    main()
