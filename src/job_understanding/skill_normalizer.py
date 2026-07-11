"""
Skill Normalization layer.

    LLMExtractor
        |
        v
    Raw skills (ExtractionResult.required_skills / preferred_skills / tech_stack —
    free-form strings straight out of the model, e.g. "Python programming",
    "PyTorch framework", "Tensor Flow")
        |
        v
    SkillNormalizer   (this file)
        |
        v
    Normalized skills (NormalizedSkill: name + category, deduplicated)

Deliberately independent of the rest of this package: SkillNormalizer
takes a list[str] in and returns a list[NormalizedSkill] out, nothing
else — no imports from and no calls into rule_based.py, llm.py, or
router.py. A caller converts ExtractionResult's raw skill lists into
normalized skills at whatever seam fits its own storage layer — this
module doesn't assume one exists.

Purely deterministic string normalization — no network call, no LLM
call of its own, no external taxonomy API. This is intentionally NOT a
full skills ontology: a small alias table for common LLM phrasing
noise, a coarse category guess for a modest set of well-known skills,
and a safe "other" fallback for everything else. Good enough for future
capabilities (matching, filtering, dashboards) to have consistent
structured data to work with — not a complete solve, and not meant to
be one yet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = ["SkillCategory", "NormalizedSkill", "SkillNormalizer"]


class SkillCategory(str, Enum):
    PROGRAMMING_LANGUAGE = "programming_language"
    ML_FRAMEWORK = "ml_framework"
    CLOUD = "cloud"
    DATABASE = "database"
    TOOL = "tool"
    CONCEPT = "concept"
    OTHER = "other"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class NormalizedSkill:
    name: str
    category: SkillCategory


# Canonical skill table: lowercase key -> (display name, category).
# Deliberately a short, hand-picked list of common skills likely to show
# up in ML/software job postings (see catalog.py for the separate,
# unrelated skill-match keyword list used by extraction) — not an
# attempt at an exhaustive taxonomy.
_CANONICAL_SKILLS: dict[str, tuple[str, SkillCategory]] = {
    "python": ("Python", SkillCategory.PROGRAMMING_LANGUAGE),
    "java": ("Java", SkillCategory.PROGRAMMING_LANGUAGE),
    "javascript": ("JavaScript", SkillCategory.PROGRAMMING_LANGUAGE),
    "typescript": ("TypeScript", SkillCategory.PROGRAMMING_LANGUAGE),
    "c++": ("C++", SkillCategory.PROGRAMMING_LANGUAGE),
    "c#": ("C#", SkillCategory.PROGRAMMING_LANGUAGE),
    "go": ("Go", SkillCategory.PROGRAMMING_LANGUAGE),
    "golang": ("Go", SkillCategory.PROGRAMMING_LANGUAGE),
    "rust": ("Rust", SkillCategory.PROGRAMMING_LANGUAGE),
    "sql": ("SQL", SkillCategory.PROGRAMMING_LANGUAGE),
    "pytorch": ("PyTorch", SkillCategory.ML_FRAMEWORK),
    "tensorflow": ("TensorFlow", SkillCategory.ML_FRAMEWORK),
    "keras": ("Keras", SkillCategory.ML_FRAMEWORK),
    "scikit-learn": ("scikit-learn", SkillCategory.ML_FRAMEWORK),
    "sklearn": ("scikit-learn", SkillCategory.ML_FRAMEWORK),
    "aws": ("AWS", SkillCategory.CLOUD),
    "gcp": ("GCP", SkillCategory.CLOUD),
    "azure": ("Azure", SkillCategory.CLOUD),
    "docker": ("Docker", SkillCategory.TOOL),
    "kubernetes": ("Kubernetes", SkillCategory.TOOL),
    "k8s": ("Kubernetes", SkillCategory.TOOL),
    "git": ("Git", SkillCategory.TOOL),
    "postgresql": ("PostgreSQL", SkillCategory.DATABASE),
    "postgres": ("PostgreSQL", SkillCategory.DATABASE),
    "mysql": ("MySQL", SkillCategory.DATABASE),
    "mongodb": ("MongoDB", SkillCategory.DATABASE),
    "redis": ("Redis", SkillCategory.DATABASE),
    "llm": ("LLM", SkillCategory.CONCEPT),
    "llms": ("LLM", SkillCategory.CONCEPT),
    "machine learning": ("Machine Learning", SkillCategory.CONCEPT),
    "deep learning": ("Deep Learning", SkillCategory.CONCEPT),
    "nlp": ("NLP", SkillCategory.CONCEPT),
}

# Common LLM phrasing noise -> the canonical key above it should resolve
# to. Whitespace/case are already normalized before this lookup runs, so
# every key here is lowercase with single spaces.
_ALIASES: dict[str, str] = {
    "python programming": "python",
    "python language": "python",
    "python programming language": "python",
    "pytorch framework": "pytorch",
    "tensor flow": "tensorflow",
    "tensorflow framework": "tensorflow",
    "deep learning models": "deep learning",
    "machine learning models": "machine learning",
    "large language model": "llm",
    "large language models": "llm",
    "natural language processing": "nlp",
}

_WHITESPACE = re.compile(r"\s+")


def _clean(raw: str) -> str:
    return _WHITESPACE.sub(" ", raw.strip())


class SkillNormalizer:
    """
    Stateless, deterministic. normalize() never raises on malformed
    input (non-string items, blank strings) — it drops them, the same
    "never crash on messy upstream data" contract as the LLM-extractor
    cleaning helpers upstream of this.
    """

    def normalize(self, raw_skills: list[Any]) -> list[NormalizedSkill]:
        seen: dict[str, NormalizedSkill] = {}
        for raw in raw_skills or []:
            if not isinstance(raw, str):
                continue
            cleaned = _clean(raw)
            if not cleaned:
                continue

            key = cleaned.lower()
            key = _ALIASES.get(key, key)

            if key in _CANONICAL_SKILLS:
                name, category = _CANONICAL_SKILLS[key]
            else:
                name, category = cleaned, SkillCategory.OTHER

            dedupe_key = name.lower()
            if dedupe_key not in seen:
                seen[dedupe_key] = NormalizedSkill(name=name, category=category)

        return list(seen.values())
