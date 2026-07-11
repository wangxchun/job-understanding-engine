"""
Regression tests against the evidence-driven fixture corpus.

Each fixture in tests/fixtures/ is derived from a real production
extraction failure (identifying details anonymized — see
docs/regression-corpus.md) and carries its own `expected` block of
confirmed-correct field values. A `null` expected value means that
field is deliberately not asserted — see the fixture's own
`provenance.notes`/`known_open_issues` for why.

One shared loader/assertion helper, not N copies of the same
assertions — every fixture goes through the exact same real entry
point (RuleBasedExtractor.extract_from_text()), no extraction logic
reimplemented here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from job_understanding import RuleBasedExtractor

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "tests" / "fixtures"

FIXTURE_NAMES = [
    "company_overview_truncation",
    "responsibilities_header_gap",
    "requirements_boundary_leak",
]


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _assert_fixture_matches(fixture_name: str) -> None:
    fixture = _load_fixture(fixture_name)
    result = RuleBasedExtractor().extract_from_text(
        fixture["text"],
        fixture.get("source_url"),
        structured_hints=fixture.get("structured_data"),
    )
    expected = fixture["expected"]
    actual = {
        "title": result.job.title,
        "company": result.job.company,
        "location": result.job.location,
        "workplace_type": result.workplace_type,
        "company_overview": result.company_overview,
        "summary": result.summary,
        "responsibilities": result.responsibilities,
        "requirements": result.requirements,
        "required_skills": result.required_skills,
        "preferred_skills": result.preferred_skills,
    }
    for field, expected_value in expected.items():
        if expected_value is None:
            continue  # not asserted — see the fixture's own provenance notes for why
        assert actual[field] == expected_value, (
            f"{fixture_name}.json: {field} mismatch — expected {expected_value!r}, got {actual[field]!r}"
        )


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_fixture_matches_expected(fixture_name: str) -> None:
    _assert_fixture_matches(fixture_name)


def test_fixture_provenance_is_honest_about_derivation() -> None:
    """Every fixture must state it's derived from a real failure, not
    invented from scratch — the corpus's core honesty guarantee
    (docs/regression-corpus.md)."""
    for name in FIXTURE_NAMES:
        fixture = _load_fixture(name)
        assert fixture["provenance"]["source"] == "anonymized_real_case", name
        assert "real production" in fixture["provenance"]["notes"], name


def test_fixture_text_contains_no_placeholder_identifiers() -> None:
    """Guards against accidentally reintroducing a real company name,
    job ID, or tracking URL into a fixture during a future edit."""
    banned_substrings = ["linkedin.com/jobs/view", "trackingId", "talent@"]
    for name in FIXTURE_NAMES:
        fixture = _load_fixture(name)
        text_lower = fixture["text"].lower()
        for banned in banned_substrings:
            assert banned.lower() not in text_lower, f"{name}.json leaked {banned!r}"
        assert fixture["source_url"] is None, name


def test_understanding_extracts_responsibilities_section() -> None:
    text = (
        "Data Engineer\n\nAcme Corp\n\nRemote\n\n"
        "Responsibilities\n"
        "Build data pipelines\n"
        "Maintain the warehouse\n"
        "On-call rotation\n"
    )
    result = RuleBasedExtractor().extract_from_text(text)
    assert result.responsibilities == [
        "Build data pipelines",
        "Maintain the warehouse",
        "On-call rotation",
    ]


def test_understanding_extracts_requirements_section() -> None:
    text = (
        "Data Engineer\n\nAcme Corp\n\nRemote\n\n"
        "Requirements\n"
        "5+ years of Python experience\n"
        "Experience with SQL\n"
    )
    result = RuleBasedExtractor().extract_from_text(text)
    assert result.requirements == [
        "5+ years of Python experience",
        "Experience with SQL",
    ]


def test_understanding_splits_required_and_preferred_catalog_skills() -> None:
    text = (
        "ML Engineer\n\nAcme Corp\n\nRemote\n\n"
        "Requirements\n"
        "Python and AWS experience required\n"
        "Preferred\n"
        "Docker experience is a plus\n"
    )
    result = RuleBasedExtractor().extract_from_text(text)
    assert "Python" in result.required_skills
    assert "AWS" in result.required_skills
    assert "Docker" in result.preferred_skills


def test_understanding_absent_when_no_recognizable_sections_or_skills() -> None:
    text = "Barista\n\nAcme Cafe\n\nDowntown\n\nWe make coffee. Join our team.\n"
    result = RuleBasedExtractor().extract_from_text(text)
    assert result.responsibilities == []
    assert result.requirements == []
    assert result.required_skills == []
    assert result.preferred_skills == []


def test_company_overview_extracted_from_about_us_section() -> None:
    text = (
        "ML Engineer\n\nAcme Corp\n\nRemote\n\n"
        "About Us\n"
        "Acme Corp builds tools that help teams ship software faster.\n\n"
        "Requirements\nPython\n"
    )
    result = RuleBasedExtractor().extract_from_text(text)
    assert result.company_overview == "Acme Corp builds tools that help teams ship software faster."


def test_company_overview_is_none_when_no_about_us_section_present() -> None:
    text = "ML Engineer\n\nAcme Corp\n\nRemote\n\nRequirements\nPython\n"
    result = RuleBasedExtractor().extract_from_text(text)
    assert result.company_overview is None
