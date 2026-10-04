"""Shortcut for ``trading-lab report`` (kept for convenience; see ``trading-lab report --help``)."""

import sys

from trading_lab.cli import main

if __name__ == "__main__":
    sys.exit(main(["report", *sys.argv[1:]]))
