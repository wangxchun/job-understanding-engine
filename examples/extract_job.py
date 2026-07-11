"""
Minimal usage example: extract structured fields from a job posting.

    python examples/extract_job.py examples/sample_posting.txt
"""
from __future__ import annotations

import sys

from job_understanding import ExtractorRouter, LLMExtractor, RuleBasedExtractor


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path-to-posting.txt>")
        raise SystemExit(1)

    text = open(sys.argv[1], encoding="utf-8").read()

    extractor = ExtractorRouter(primary=LLMExtractor(), fallback=RuleBasedExtractor())
    result = extractor.extract_from_text(text)

    print(f"Title:            {result.job.title}")
    print(f"Company:          {result.job.company}")
    print(f"Location:         {result.job.location}")
    print(f"Workplace type:   {result.workplace_type}")
    print(f"Title confidence: {result.title_confidence}")
    print()
    print(f"Company overview: {result.company_overview}")
    print(f"Summary:          {result.summary}")
    print()
    print("Responsibilities:")
    for item in result.responsibilities:
        print(f"  - {item}")
    print()
    print("Requirements:")
    for item in result.requirements:
        print(f"  - {item}")
    print()
    print(f"Required skills:  {result.required_skills}")
    print(f"Preferred skills: {result.preferred_skills}")


if __name__ == "__main__":
    main()
