"""Allow ``python -m trading_lab ...``."""

import sys

from trading_lab.cli import main

sys.exit(main())
