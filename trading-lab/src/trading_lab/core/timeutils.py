"""Time helpers. The simulation never reads the wall clock; callers supply timestamps."""

from datetime import datetime, timezone


def ensure_utc(value: datetime, name: str = "timestamp") -> datetime:
    """Return ``value`` converted to UTC, rejecting naive datetimes."""
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware (UTC); got naive {value!r}")
    return value.astimezone(timezone.utc)
