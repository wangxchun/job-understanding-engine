"""
Provider-agnostic extraction interface.

JobExtractor is the seam between "what a caller needs" (a Job, plus how
much to trust the title, plus structured understanding fields) and "how
it's produced" (regex/heuristics — RuleBasedExtractor; an LLM call —
LLMExtractor). Anything downstream of extraction depends only on this
interface and ExtractionResult — never on a concrete implementation.

    Caller (CLI / API / any ingestion client)
              |
              v
         JobExtractor  (this file)
              |
       ----------------
       |              |
    RuleBased     LLMExtractor

Both extractors implement extract_from_text()/extract_from_url() with
the same signatures — a caller never knows or needs to know which one
actually ran (see router.py for how routing between them works).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .schema import Job

__all__ = ["JobExtractor", "ExtractionResult", "TitleConfidence", "JobFetchError"]


class TitleConfidence(str, Enum):
    """
    How much an extractor trusts its own title guess. This is the seam
    for human-in-the-loop confirmation: a caller can auto-accept HIGH/
    MEDIUM guesses but should ask a human to confirm/correct on LOW.

    HIGH   - an explicit field ("Title:"), structured source data (e.g.
             JSON-LD), or a strong, unambiguous prose pattern ("As a
             Machine Learning Engineer, ...").
    MEDIUM - a standalone heading line that plausibly reads as a title
             (contains a role keyword like "Engineer"/"Manager"), or
             unstructured page metadata (Open Graph / <title>).
    LOW    - a generic best-effort guess with no real positive signal —
             the exact failure mode that once produced "Why Choose
             Corsearch?" or "global leader in Trademark and Brand
             Protection" as a "title". Never persisted without a human
             confirming/overriding it.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def __str__(self) -> str:
        return self.value


class JobFetchError(RuntimeError):
    """
    Raised whenever a job page can't be turned into a Job — bad URL,
    network failure, or a site blocking/gating the page. `reason` is
    always one of "login required", "blocked request", or "unavailable
    page" so callers can show a consistent, friendly message instead of
    a traceback. Shared across extractor implementations — an LLM-backed
    extractor that also fetches a URL raises the same exception shape.
    """

    def __init__(self, url: str, reason: str, detail: str = "") -> None:
        self.url = url
        self.reason = reason
        self.detail = detail
        message = f"{reason}: {url}"
        if detail:
            message += f" ({detail})"
        super().__init__(message)


@dataclass
class ExtractionResult:
    """
    What every JobExtractor method returns: a `Job`, unchanged, plus
    title confidence (the human-in-the-loop trigger) and the structured
    "job understanding" fields — the actual output this whole engine
    exists to produce.

    `department`/`summary`/`seniority`/`employment_type`/
    `required_skills`/`preferred_skills`/`tech_stack` are understanding
    fields an LLM-backed extractor can populate from structured output.
    `RuleBasedExtractor` derives most of these from generic section-
    header scanning; a field it can't derive simply stays at its
    default (`None`/empty list) — "not extracted", never an error.

    `responsibilities`/`requirements` are day-to-day duties and stated
    qualifications, each as a list of short bullet-style strings (not
    one freeform paragraph), so a downstream consumer never has to
    re-parse prose to get structured items.

    `education_requirements` is a structured subset of `requirements`
    (degree/certification bullets), kept separate for direct display
    without being a replacement for the fuller list.

    `company_overview` is a short "what this company does" summary —
    drawn ONLY from the posting's own "About Us"/"Who We Are"-style
    text when present, never an external lookup or invented from
    outside knowledge (see `rule_based.py`'s `_derive_company_overview()`
    and `llm.py`'s system prompt, both of which enforce this explicitly).

    `workplace_type`/`company_url` are sourced only from a posting's own
    schema.org `JobPosting` JSON-LD, never guessed from prose text.
    """

    job: Job
    title_confidence: TitleConfidence
    department: Optional[str] = None
    summary: Optional[str] = None
    seniority: Optional[str] = None
    employment_type: Optional[str] = None
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    tech_stack: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    education_requirements: list[str] = field(default_factory=list)
    company_overview: Optional[str] = None
    workplace_type: Optional[str] = None
    company_url: Optional[str] = None


class JobExtractor(ABC):
    """Interface every extraction implementation (rule-based, LLM-based,
    ...) must satisfy."""

    @abstractmethod
    def extract_from_text(
        self, text: str, source_url: Optional[str] = None, *, structured_hints: Optional[dict] = None,
    ) -> ExtractionResult:
        """Turn pasted/captured job-description text into an ExtractionResult.

        `structured_hints` is optional raw page metadata an ingestion
        client already had for free — `{"json_ld": [...], "meta": {...}}`
        — that an implementation may use to improve title/company/
        location/description accuracy over line-based text guessing
        alone. `None` (the default) means "no hints available" — every
        implementation must behave identically to when hints exist,
        just with a narrower signal set to work from.
        """
        ...

    @abstractmethod
    def extract_from_url(self, url: str) -> ExtractionResult:
        """Fetch a job posting URL and turn it into an ExtractionResult."""
        ...
