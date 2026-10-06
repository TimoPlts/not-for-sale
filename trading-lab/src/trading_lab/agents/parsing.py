"""Prompt instructions and strict parsing of JSON answers (for LLM-backed agents)."""

from __future__ import annotations

import json
import re
from typing import Mapping, Sequence

from trading_lab.agents.base import AgentResponse, AgentResponseError

RESPONSE_INSTRUCTIONS = """\
You are a cautious crypto market analyst taking part in a paper-trading research simulation.
You can only recommend; you cannot place orders. The portfolio is long-only.
Reply with a single JSON object and nothing else:
{"direction": "buy" | "sell" | "hold", "confidence": <number between 0 and 1>, "rationale": "<one or two sentences>"}
"buy" means open a long position, "sell" means close an existing long, "hold" means do nothing.
Use "hold" with confidence 0 when the situation is unclear."""

_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_agent_json(text: str, labels: Mapping[str, Sequence[str]] | None = None) -> AgentResponse:
    """Parse ``{"direction", "confidence", "rationale"}`` from model output.

    The JSON may be wrapped in a ```json fence, or surrounded by text as long
    as exactly one top-level object is present. Anything else raises
    ``AgentResponseError``. Nothing is guessed or repaired.

    ``labels`` names extra required fields and their allowed values, e.g.
    ``{"regime": ("bullish_trend", "sideways", ...)}``. They are matched
    case-insensitively and returned in ``AgentResponse.extra``.
    """
    if not isinstance(text, str) or not text.strip():
        raise AgentResponseError("empty response")
    fenced = _FENCED.search(text)
    candidate = fenced.group(1) if fenced else text.strip()
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise AgentResponseError("no JSON object in response")
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise AgentResponseError(f"invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise AgentResponseError("response JSON must be an object")
    required = {"direction", "confidence", *(labels or {})}
    missing = required - set(data)
    if missing:
        raise AgentResponseError(f"response is missing {sorted(missing)}")
    direction = data["direction"]
    if isinstance(direction, str):
        direction = direction.strip().lower()
    extra: dict[str, str] = {}
    for field, allowed in (labels or {}).items():
        value = data[field]
        normalized = value.strip().lower() if isinstance(value, str) else value
        if normalized not in allowed:
            raise AgentResponseError(f"{field} must be one of {list(allowed)}, got {value!r}")
        extra[field] = normalized
    return AgentResponse(direction, data["confidence"], str(data.get("rationale", "")), extra)
