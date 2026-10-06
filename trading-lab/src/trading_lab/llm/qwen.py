"""Qwen, through an OpenAI-compatible chat-completions endpoint.

Environment variables (never config, never committed):

* ``QWEN_API_URL``: base URL such as ``https://host/v1``, or the full
  ``https://host/v1/chat/completions`` URL
* ``QWEN_API_KEY``: bearer token for that endpoint
* ``QWEN_MODEL``: model name, e.g. ``qwen2.5-7b-instruct``
"""

from __future__ import annotations

from trading_lab.llm.openai_compat import OpenAICompatibleProvider


class QwenProvider(OpenAICompatibleProvider):
    name = "qwen"
    env_prefix = "QWEN"
