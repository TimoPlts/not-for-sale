"""Shortcut for ``trading-lab backtest`` (kept for convenience; see ``trading-lab backtest --help``)."""

import sys

from trading_lab.cli import main

if __name__ == "__main__":
    sys.exit(main(["backtest", *sys.argv[1:]]))
