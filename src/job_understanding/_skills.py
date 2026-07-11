from __future__ import annotations

import re

__all__ = ["build_skill_engine"]


def build_skill_engine(keywords: list[str]) -> tuple[re.Pattern, dict[str, str]]:
    """
    Build a regex matcher and canonical-name map from a keyword list.

    Keywords may contain regex escapes (e.g. "C\\+\\+" for C++); these are
    unescaped to produce the display name in the canonical map.

    Returns:
        pattern   — case-insensitive regex matching any keyword at word boundaries
        canonical — maps lowercase match → display name (e.g. "pytorch" → "PyTorch")
    """
    pattern = re.compile(
        r"\b(" + "|".join(keywords) + r")\b",
        re.IGNORECASE,
    )
    canonical: dict[str, str] = {}
    for kw in keywords:
        display = re.sub(r"\\(.)", r"\1", kw)  # unescape: C\+\+ → C++
        canonical[display.lower()] = display
    return pattern, canonical
