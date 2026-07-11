"""
Job Understanding Engine — turns noisy job posting text into structured,
evidence-grounded hiring intelligence.

    text/URL
        |
        v
    JobExtractor  (RuleBasedExtractor | LLMExtractor | ExtractorRouter)
        |
        v
    ExtractionResult

See docs/architecture.md for the full design, docs/design-decisions.md
for the reasoning behind specific choices, and docs/regression-corpus.md
for how correctness is evaluated against real extraction failures.
"""
from .base import ExtractionResult, JobExtractor, JobFetchError, TitleConfidence
from .llm import LLMExtractionError, LLMExtractor
from .router import ExtractorRouter
from .rule_based import RuleBasedExtractor
from .schema import Job
from .skill_normalizer import NormalizedSkill, SkillCategory, SkillNormalizer

__all__ = [
    "Job",
    "JobExtractor",
    "ExtractionResult",
    "TitleConfidence",
    "JobFetchError",
    "RuleBasedExtractor",
    "LLMExtractor",
    "LLMExtractionError",
    "ExtractorRouter",
    "SkillNormalizer",
    "NormalizedSkill",
    "SkillCategory",
]
