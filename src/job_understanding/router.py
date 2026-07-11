"""
ExtractorRouter — the "auto" strategy: LLM first, rule-based fallback.

    ExtractorRouter
            |
     ---------------------
     |                   |
LLMExtractor       RuleBasedExtractor

Tries `primary` (normally LLMExtractor) first; falls back to `fallback`
(normally RuleBasedExtractor) whenever the primary's result isn't
trustworthy (title_confidence == LOW) or the primary raises
LLMExtractionError outright (missing `anthropic` package, no credentials,
API failure — llm.py wraps all of these into LLMExtractionError, so
catching that one type covers "never crash" for all three).

Deliberately does NOT catch anything else. In particular, JobFetchError
(the URL couldn't be fetched at all) is allowed to propagate rather than
triggering a fallback: both extractors would fetch the exact same URL via
the exact same rule_based.fetch_job_page(), so retrying with the other
extractor can't succeed where the first one failed — it would just be a
second identical network failure. A caller handling JobFetchError for a
single extractor doesn't need anything extra here.

Implements the same JobExtractor interface as its two children, so a
caller building extractors through this router never needs to know
routing is happening — it only ever sees a JobExtractor.
"""
from __future__ import annotations

from typing import Optional

from .base import ExtractionResult, JobExtractor, TitleConfidence
from .llm import LLMExtractionError

__all__ = ["ExtractorRouter"]

# The confidence levels the router trusts enough to skip the fallback.
_ACCEPTABLE = (TitleConfidence.HIGH, TitleConfidence.MEDIUM)


class ExtractorRouter(JobExtractor):
    """Auto-fallback JobExtractor: LLM first, rule-based second."""

    def __init__(self, primary: JobExtractor, fallback: JobExtractor) -> None:
        self.primary = primary
        self.fallback = fallback

    def extract_from_text(
        self, text: str, source_url: Optional[str] = None, *, structured_hints: Optional[dict] = None,
    ) -> ExtractionResult:
        result = self._try_primary(
            lambda: self.primary.extract_from_text(text, source_url, structured_hints=structured_hints)
        )
        if result is not None:
            return result
        return self.fallback.extract_from_text(text, source_url, structured_hints=structured_hints)

    def extract_from_url(self, url: str) -> ExtractionResult:
        result = self._try_primary(lambda: self.primary.extract_from_url(url))
        if result is not None:
            return result
        return self.fallback.extract_from_url(url)

    def _try_primary(self, call) -> Optional[ExtractionResult]:
        """Run `call`; return its ExtractionResult if trustworthy, else
        None to signal "use the fallback" — for either reason (LOW
        confidence, or the primary backend being unusable)."""
        try:
            result = call()
        except LLMExtractionError:
            return None

        if result.title_confidence in _ACCEPTABLE:
            return result
        return None
