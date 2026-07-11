"""
Minimal configuration for the LLM-backed extractor.

`LLMExtractor` needs exactly two knobs: which API key to use (if any —
absent, the Anthropic SDK resolves credentials from the environment on
its own) and which model to call. Everything else a full application
config might have (storage paths, backend selection, other services'
credentials) belongs to the caller, not to this engine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["Settings", "get_settings"]


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: Optional[str] = field(default=None, repr=False)
    llm_model: str = "claude-opus-4-8"


def get_settings() -> Settings:
    """Read settings from the environment. No caching — cheap enough to
    call per-extractor, and tests can freely monkeypatch os.environ."""
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        llm_model=os.getenv("JOB_UNDERSTANDING_LLM_MODEL", "claude-opus-4-8"),
    )
