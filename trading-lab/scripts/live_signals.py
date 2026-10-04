"""Shortcut for ``trading-lab signals`` (kept for convenience; see ``trading-lab signals --help``)."""

import sys

from trading_lab.cli import main

if __name__ == "__main__":
    sys.exit(main(["signals", *sys.argv[1:]]))
