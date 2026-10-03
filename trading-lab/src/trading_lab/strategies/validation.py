"""Parameter validation helpers for strategy constructors."""

from __future__ import annotations

import math


def check_int(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def check_range(
    name: str, value: object, low: float, high: float, *, inclusive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    ok = low <= value <= high if inclusive else low < value < high
    if not ok:
        raise ValueError(f"{name} must be in ({low}, {high}), got {value!r}")
    return float(value)
