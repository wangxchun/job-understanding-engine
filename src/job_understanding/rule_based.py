"""
Deterministic, regex/heuristic JobExtractor implementation — no LLM, no
scraping beyond plain HTTP + stdlib HTML parsing.

Sits behind the same JobExtractor interface as LLMExtractor (see
base.JobExtractor) so a caller can plug in an LLM-backed extractor
without touching anything here. extract_from_text() and
extract_from_url() are built to give equally good results for the same
underlying data, regardless of which one a caller happens to use:

  1. Structured metadata (title/company/location). extract_from_url()
     parses a fetched page's schema.org JobPosting JSON-LD — see
     _find_jobposting()/_company_from_jsonld()/etc. below.
     extract_from_text() accepts the same kind of data via
     `structured_hints` (a caller that already has a rendered page's
     JSON-LD — e.g. a browser-based ingestion client — can forward it,
     since a raw-text-only extractor never has it on its own) and
     reuses those exact same functions to interpret it — see
     _parse_structured_hints() below.
  2. Full Job Understanding (summary/responsibilities/requirements/
     required_skills/preferred_skills) from plain description text —
     see _extract_understanding() below. Deliberately NOT a
     reimplementation of what llm.py already does well from an LLM; a
     generic section-header scan (Responsibilities/Requirements/
     Preferred Qualifications, several common spellings each — same
     "structural pattern, not per-site markup" shape as
     _guess_structured_header()/_NOISE_HEADERS) plus the existing skill
     catalog (catalog.py + _skills.py's build_skill_engine()) — reused,
     not duplicated. This only closes the gap for the zero-cost
     rule-based path; a caller routing through ExtractorRouter already
     gets richer results from LLMExtractor the same way it always has.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, NamedTuple, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .schema import Job
from ._skills import build_skill_engine

from .catalog import TRACKED_SKILLS
from .base import (
    ExtractionResult, JobExtractor, JobFetchError, TitleConfidence,
)

__all__ = ["RuleBasedExtractor", "fetch_job_page", "looks_like_url", "describe_source"]

_COMPANY_PREFIX = re.compile(r"(?i)^\s*company\s*[:\-]\s*(.+)$")
_LOCATION_PREFIX = re.compile(r"(?i)^\s*location\s*[:\-]\s*(.+)$")
_TITLE_PREFIX = re.compile(r"(?i)^\s*(?:job\s*)?title\s*[:\-]\s*(.+)$")
_AT_COMPANY = re.compile(r"(?i)\bat\s+([A-Z][\w&.,'\-]{1,60})\s*$")
_LOCATION_HINT = re.compile(r"(?i)\b(remote|hybrid|on[- ]?site)\b")
_CITY_COMMA = re.compile(r"^[A-Z][\w.\- ]{1,40},\s*[A-Za-z .]{2,40}$")
# "Amsterdam, North Holland, Netherlands" — LinkedIn's 3-part city/region/
# country header shape, distinct from the 2-part "City, Country" above.
_CITY_REGION_COUNTRY = re.compile(
    r"^[A-Z][\w.\- ]{1,40},\s*[A-Za-z .]{2,40},\s*[A-Za-z .]{2,40}$"
)
# Sprint 15.8.4: "<City> Area" / "Greater <City> Area" / "<City>
# Metropolitan Area" / "<City> Bay Area" — a generic geographic-naming
# convention many job boards use (not LinkedIn-specific), distinct from
# the comma-separated shapes above. [A-Z] stays genuinely case-
# sensitive for the place-name tokens (only the fixed keywords
# Greater/Metropolitan/Bay/Area are case-alternated) — same "must look
# like a proper noun" reasoning _AS_A_ROLE's own comment documents.
_AREA_LOCATION = re.compile(
    r"^(?:[Gg]reater\s+)?[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)*\s+"
    r"(?:[Mm]etropolitan\s+|[Bb]ay\s+)?[Aa]rea$"
)
# Sprint 15.8.4: a STANDALONE workplace-mode line ("Hybrid" and nothing
# else) — deliberately an exact, whole-line match, not the substring
# search _LOCATION_HINT above does. That distinction is the actual fix
# for "location = Hybrid · 2 weeks ago · 87 applicants": a loose
# substring search matches a compound metadata line just because it
# contains the word "hybrid" somewhere in it; this only matches when
# the ENTIRE line is nothing but the workplace-mode word itself.
_WORKPLACE_TYPE_LINE = re.compile(r"(?i)^(remote|hybrid|on-?site|in-?office)\s*$")

# Generic section headers LinkedIn/Greenhouse/etc. postings commonly open
# with — never a job title, but structurally indistinguishable from one
# (short, no punctuation) without this denylist.
_NOISE_HEADERS = {
    "about the job", "about the role", "about this role", "job description",
    "overview", "responsibilities", "responsibilities and duties",
    "requirements", "qualifications", "about us", "about the company",
    "company overview", "who we are", "the role", "the opportunity",
    "what you'll do", "benefits", "description", "summary",
    # Sprint 21 Round 5: "additional details" — a real, corpus-evidenced
    # trailing boundary header (a real anonymized production case, see docs/regression-corpus.md), introducing EEO/accommodation
    # boilerplate at the end of a posting. Generic, company-agnostic
    # phrasing, the same shape as every other entry in this set — not
    # itself a responsibilities/requirements bucket, just a stop marker,
    # which is exactly what was missing: without it, _collect_section_items()
    # ran straight past the last real requirement bullet and swallowed
    # this header's own text as a bogus final item. See Sprint 21 Round 4's
    # documented, deliberately-deferred finding and Round 5's confirming
    # non-destructive hypothesis test (docs/regression-corpus.md) — adding it here affects only where collection
    # stops, never what gets included.
    "additional details",
}

# Sprint 15.8.2: a section-boundary hint, not a new denylist — the
# subset of _NOISE_HEADERS that INTRODUCES the real posting body rather
# than ending an intro (unlike "Responsibilities"/"Requirements", which
# mark where the intro *stops*). "About the job" is the single most
# common LinkedIn-style header of this shape: a real posting rarely
# writes any prose worth summarizing before it, and _derive_summary()
# (below) previously only ever looked at text BEFORE the first
# recognized header — meaning it never found anything to summarize on
# a posting shaped "chrome, About the job, real intro, Responsibilities,
# ...", exactly the sprint's own reported example. See
# _derive_summary()'s docstring for how this is used.
_INTRO_HEADERS = {
    "about the job", "about the role", "about this role", "job description",
    "overview", "role overview", "position overview", "the opportunity",
    "description", "summary",
}

# Section headers whose text incorporates the company name (so a plain
# denylist can't catch them), e.g. "Why Choose Corsearch?", "Why Work at
# Corsearch" — matched structurally instead of by exact string.
_NOISE_HEADER_PATTERNS = (
    re.compile(r"(?i)^why\s+(choose|work\s+(for|at))\b"),
)

# A bare "Label:" line with nothing after the colon — the value is on the
# next line instead (common when copy-pasting a rendered definition list).
_LABEL_LINE = re.compile(r"(?i)^[\w][\w /]*:\s*$")

# The strongest positive signal that a heading line is a job title, short
# of an explicit "Title:" field — most role titles contain one of these.
#
# Sprint 15.8.8: "researcher" added — a real, reproduced failure ("AI /
# Machine Learning Researcher" containing no other keyword) fell through
# to tier 4's best-effort heading guess, which grabbed an unrelated
# earlier heading-shaped line (the company name) instead; separately,
# the weak line-2 company fallback's own role-keyword guard
# (_guess_company(), below) failed to reject "AI / Machine Learning
# Researcher" as a company candidate for the exact same reason, since
# it reuses this same list — one missing keyword produced both halves
# of a title/company swap. Not a per-posting special case: "researcher"
# is as generic a role keyword as "scientist"/"analyst" already here.
_TITLE_KEYWORDS = re.compile(
    r"(?i)\b(engineer|manager|scientist|analyst|developer|designer|"
    r"specialist|director|lead|architect|consultant|coordinator|"
    r"intern|associate|administrator|officer|recruiter|owner|researcher)\b"
)

# "As an ML Engineer you'll ..." / "As a Machine Learning Engineer, ..." —
# captures the role name, stopping at the first clause boundary rather
# than swallowing the rest of the sentence. The "within/in/for our/the ...
# team" stop is what keeps a trailing department clause (see
# _DEPARTMENT_CLAUSE below) out of the title — e.g. "As a Machine
# Learning Engineer within our Advertising team, you'll..." must yield
# title "Machine Learning Engineer", not "...within our Advertising team".
#
# Deliberately NOT `(?i)` overall: under re.IGNORECASE, `[A-Z]` matches
# lowercase letters too, which silently defeated the "must look like a
# proper noun" check and let sentences like "As a global leader in
# Trademark and Brand Protection..." through (a real bug found in
# production — "global" matched `[A-Z]` case-insensitively). Only the
# fixed keyword literals are case-alternated; the capture group's leading
# `[A-Z]` stays genuinely case-sensitive.
_DEPARTMENT_STOP = r"\s+(?:within|in|for)\s+(?:our|the)\b"
_AS_A_ROLE = re.compile(
    r"\b[Aa]s\s+(?:[Aa]n?|[Tt]he)\s+([A-Z][A-Za-z0-9&/+\-\s]{2,60}?)"
    r"(?=[,\.]|\s+you\b|\s+who\b|\s+we\b|\s+responsible\b|" + _DEPARTMENT_STOP + r")"
)

# "We are looking for a Machine Learning Engineer ..." / "...seeking an
# ML Engineer..." / "...hiring a Data Scientist...".
_LOOKING_FOR_ROLE = re.compile(
    r"\b[Ww]e(?:'re|\s+are)\s+(?:currently\s+)?(?:looking|seeking|hiring)"
    r"\s+for\s+(?:[Aa]n?|[Tt]he)\s+([A-Z][A-Za-z0-9&/+\-\s]{2,60}?)"
    r"(?=[,\.]|\s+who\b|\s+to\b|\s+that\b|" + _DEPARTMENT_STOP + r")"
)

# "within our Advertising team" / "in the Advertising team" / "for our
# Advertising organization" — the department/team clause that must be
# kept OUT of the title (see _DEPARTMENT_STOP above) and instead becomes
# its own field.
_DEPARTMENT_CLAUSE = re.compile(
    r"(?i)\b(?:within|in|for)\s+(?:our|the)\s+([A-Z][\w &/]{1,40}?)\s+"
    r"(?:team|organization|org|group|department)\b"
)

# "Staff ML Engineer - Advertising" — LinkedIn's own page header renders
# title and department on one line, separated by a dash. This is checked
# against lines[0] only (see _guess_structured_header) since it's a
# header shape, not a sentence pattern that can appear anywhere.
_HEADER_TITLE_DEPARTMENT = re.compile(r"^(.{2,60}?)\s+[-–]\s+([A-Z][\w&/,. ]{1,40})$")

# How many leading lines we're willing to scan for header fields before
# giving up and treating everything as description. Keeps the heuristic
# from misfiring deep inside a long job description.
_HEADER_WINDOW = 8

# The "As a ROLE" / "We are looking for a ROLE" sentence patterns describe
# the role in prose rather than a standalone heading line, so they can
# legitimately appear a bit further down (e.g. under a "The Role" section)
# than the header fields above — hence the wider window.
_TITLE_SENTENCE_WINDOW = 20


def _clean_lines(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def _is_noise_header(line: str) -> bool:
    normalized = line.rstrip(":?!").strip().lower()
    if normalized in _NOISE_HEADERS:
        return True
    return any(p.match(line) for p in _NOISE_HEADER_PATTERNS)


def _line_matches_title(line: str, title: str) -> bool:
    """Sprint 15.8.9: is `line` the raw text's rendering of the already-
    resolved `title` — verbatim, OR as the title portion of a "Title -
    Department" header line (_HEADER_TITLE_DEPARTMENT — the exact shape
    _guess_structured_header() already recognizes for lines[0], and the
    same shape _AS_A_ROLE's _DEPARTMENT_STOP clause already strips out
    when the title is instead derived from an "As a ROLE in our TEAM"
    sentence)?

    Root cause this fixes: a real reported LinkedIn posting renders its
    heading as "Senior Machine Learning Engineer - Applied ML &
    Research" — ONE line combining the role and the team/department.
    The title resolves correctly (via the "As a ROLE in our TEAM..."
    sentence tier, which already strips the department clause) to the
    bare "Senior Machine Learning Engineer" — but every function in
    this module that needs to find "which raw line IS the title" was
    comparing the title against a line with strict equality, and no
    line equals the bare title verbatim; the only candidate line has
    the department suffix still attached. The structural company
    signals (which locate a company relative to the title's line index)
    silently found nothing, not because the company was rejected, but
    because the title's own line could never be located at all — a
    third, distinct variant of Sprint 15.8.8's "title used for
    positional lookup isn't always a literal, bare line" bug class
    (that sprint fixed the LLM-paraphrase case; this fixes the title-
    plus-department-on-one-line case, for BOTH backends, since this
    check runs before any backend-specific behavior).

    Reuses _HEADER_TITLE_DEPARTMENT verbatim — no new pattern; this
    function only widens WHERE that existing shape is checked, from
    "lines[0] only" to "any line being tested against a resolved
    title"."""
    normalized_title = title.strip().lower()
    if line.strip().lower() == normalized_title:
        return True
    m = _HEADER_TITLE_DEPARTMENT.match(line)
    return bool(m and m.group(1).strip().lower() == normalized_title)


def _looks_like_heading(line: str) -> bool:
    """A plausible title/company heading: not a noise header, not a bare
    "Label:" line, short, and not a full sentence."""
    if not line or _is_noise_header(line) or _LABEL_LINE.match(line):
        return False
    return len(line) <= 80 and not line.endswith((".", ";"))


def _plausible_title(candidate: str) -> bool:
    """
    Guards the "As a X" / "We are looking for a X" sentence patterns
    against grabbing a company-description clause instead of a real
    title — e.g. "As a global leader in Trademark and Brand Protection,
    we have built..." matches the same sentence shape as "As a Machine
    Learning Engineer, you'll...", so the regex alone can't tell them
    apart. A real title either contains a role keyword or is short
    (job titles are rarely more than a handful of words); a company
    description clause is normally longer and keyword-free.
    """
    return bool(_TITLE_KEYWORDS.search(candidate)) or len(candidate.split()) <= 4


def _immediately_precedes_company_card(lines: list[str], index: int) -> bool:
    """Sprint 15.8.5: a job title is never immediately followed by "N
    followers"/"N employees" — that shape is a company-card signature,
    not a job-title signature. Reuses _is_company_card_anchor() (no
    new pattern). This is the root-level fix for a real, reproduced
    collision: when the extracted text leads directly into the company
    card with no separate title line before it, the heading-line title
    guesses below (tiers 3/4) had nothing better to grab than the
    company's own name — which then made _guess_company_from_card()'s
    title-repeat guard reject that exact same name as a company
    candidate later, producing "Company: Unknown" even though the name
    was sitting right there. Preventing the wrong TITLE guess in the
    first place is more robust than patching around the collision
    downstream. Checks a small window ahead (not just the very next
    line) to cover "name, Follow, count" layouts, not just "name,
    count" — stops looking further ahead as soon as a line isn't still
    card-shaped."""
    for offset in range(1, 3):
        next_index = index + offset
        if next_index >= len(lines):
            break
        next_line = lines[next_index]
        if _is_company_card_anchor(next_line):
            return True
        if next_line.strip().lower() not in _COMPANY_CARD_UI_LABELS and not _looks_like_heading(next_line):
            break
    return False


def _guess_title_with_confidence(lines: list[str]) -> tuple[str, TitleConfidence]:
    # 1. Explicit "Title:" / "Job Title:" field — HIGH.
    for line in lines[:_HEADER_WINDOW]:
        m = _TITLE_PREFIX.match(line)
        if m:
            return m.group(1).strip(), TitleConfidence.HIGH

    # 2. Strong prose patterns — "As a Machine Learning Engineer, you'll
    #    ..." or "We are looking for a ... Engineer ..." — HIGH, but only
    #    once the candidate passes _plausible_title; otherwise keep
    #    scanning rather than returning a company-description clause that
    #    happens to match the same sentence shape.
    for line in lines[:_TITLE_SENTENCE_WINDOW]:
        m = _AS_A_ROLE.search(line) or _LOOKING_FOR_ROLE.search(line)
        if m:
            candidate = m.group(1).strip()
            if _plausible_title(candidate):
                return candidate, TitleConfidence.HIGH

    # 3. A standalone heading line containing a common role keyword
    #    (Engineer, Manager, ...) — MEDIUM: a reasonable candidate, but
    #    weaker than an explicit field or a full sentence naming the role.
    #    Sprint 15.8.5: skips a line immediately followed by a company-
    #    card anchor ("N followers") — see
    #    _immediately_precedes_company_card()'s own docstring for the
    #    real collision this prevents.
    for index, line in enumerate(lines[:_HEADER_WINDOW]):
        if (
            _looks_like_heading(line) and _TITLE_KEYWORDS.search(line)
            and not _immediately_precedes_company_card(lines, index)
        ):
            return line, TitleConfidence.MEDIUM

    # 4. Best-effort: first heading-shaped line that isn't a known noise
    #    header or a bare label — LOW. No positive signal that this is
    #    actually a title rather than a company name or random section
    #    heading; the caller must not save this without confirmation.
    #    Sprint 15.8.5: same company-card-adjacency skip as tier 3.
    for index, line in enumerate(lines[:_HEADER_WINDOW]):
        if _looks_like_heading(line) and _immediately_precedes_company_card(lines, index):
            continue
        if _looks_like_heading(line):
            return line, TitleConfidence.LOW

    return (lines[0] if lines else "Untitled"), TitleConfidence.LOW


def _guess_title(lines: list[str]) -> str:
    return _guess_title_with_confidence(lines)[0]


def _guess_department(lines: list[str]) -> Optional[str]:
    """
    Priority 3 (explicit pattern in description): "within our Advertising
    team", "in the Advertising team", "for our Advertising organization".
    Independent of which title tier fired — a department clause can
    accompany an explicit "Title:" field just as easily as an "As a ..."
    sentence.
    """
    for line in lines[:_TITLE_SENTENCE_WINDOW]:
        m = _DEPARTMENT_CLAUSE.search(line)
        if m:
            return m.group(1).strip()
    return None


def _looks_like_location_line(line: str) -> bool:
    return bool(
        _CITY_REGION_COUNTRY.match(line) or _CITY_COMMA.match(line)
        or _AREA_LOCATION.match(line) or _LOCATION_HINT.search(line)
    )


def _guess_structured_header(lines: list[str]) -> Optional[tuple[str, Optional[str], str]]:
    """
    Priority 1 (structured header information): LinkedIn's own rendered
    page header — "Staff ML Engineer - Advertising" immediately followed
    by "Amsterdam, North Holland, Netherlands". This is the single most
    reliable signal available for a pasted posting (it's the site's own
    structured header, not prose to pattern-match), so it's checked
    before any Priority-3 field/sentence pattern.

    Both lines must match their expected shape — a title+department line
    on its own proves nothing; only the location line right after it
    confirms this really is a header block and not, say, a sentence that
    happens to contain a dash.

    Returns (title, department, location) or None.
    """
    if len(lines) < 2:
        return None
    m = _HEADER_TITLE_DEPARTMENT.match(lines[0])
    if not m or not _looks_like_location_line(lines[1]):
        return None
    return m.group(1).strip(), m.group(2).strip(), lines[1]


_COMPANY_STOPWORDS = {
    "this", "the", "we", "our", "it", "they", "you", "i", "as", "who",
    "what", "when", "where", "why", "how", "here", "there",
}

# "At Corsearch, we are dedicated..." — company named at the start of the
# company-intro sentence, rather than as an explicit "Company:" field or
# a trailing "... at Company" phrase (both handled separately below).
_COMPANY_AT_START = re.compile(r"(?i)^\s*at\s+([A-Z][\w&.'\-]{1,40}?)\s*,")
# "Join Corsearch as we build..." / "Join Corsearch!"
_COMPANY_JOIN = re.compile(r"(?i)^\s*join\s+([A-Z][\w&.'\-]{1,40}?)\b")
# "Corsearch is a global leader..."
_COMPANY_IS = re.compile(r"(?i)^\s*([A-Z][\w&.'\-]{1,40})\s+is\b")


def _guess_company_from_explicit_patterns(lines: list[str]) -> Optional[str]:
    """The confident tiers of company text-heuristics — an explicit
    "Company:" field, an "... at X" phrase, or a company-intro sentence
    pattern ("At X, we...", "Join X...", "X is..."). Deliberately
    excludes _guess_company()'s weak last-resort "assume line 2" guess
    below: these three sub-tiers are strong, explicit signals, while
    the line-2 guess has no real positive signal at all — appropriate
    as RuleBasedExtractor's own last resort before "Unknown" (nothing
    else to fall back to), but never something that should override a
    capable LLM's own contextual inference (see _resolve_company()'s
    `weak_fallback` parameter, Sprint 15.8.3)."""
    for line in lines[:_HEADER_WINDOW]:
        m = _COMPANY_PREFIX.match(line)
        if m:
            return m.group(1).strip()
    for line in lines[:3]:
        m = _AT_COMPANY.search(line)
        if m:
            return m.group(1).strip()
    for line in lines[:_HEADER_WINDOW]:
        for pattern in (_COMPANY_AT_START, _COMPANY_JOIN, _COMPANY_IS):
            m = pattern.match(line)
            if m:
                candidate = m.group(1).strip().rstrip(".,")
                if candidate.lower() not in _COMPANY_STOPWORDS:
                    return candidate
    return None


def _guess_company(lines: list[str]) -> str:
    explicit = _guess_company_from_explicit_patterns(lines)
    if explicit:
        return explicit
    # Best-effort fallback: LinkedIn postings commonly list the company
    # name on the second non-empty line, right after the title. Guarded
    # (Sprint 15.7) against the real failure mode this produced in
    # production: a browser client's <main>-based extraction (see
    # extension/content.js) can legitimately have line 2 be a repeat of
    # the title (a sticky header, a duplicated heading) rather than a
    # company name, with nothing above having fired to catch it. Rather
    # than guess wrong silently, this now only returns line 2 when it
    # doesn't look like a duplicate/near-duplicate of the title and
    # doesn't itself look like a role heading — otherwise "Unknown",
    # the same "not extracted, never a wrong answer" contract every
    # other unresolved field on ExtractionResult already has.
    if len(lines) > 1 and len(lines[1]) < 60:
        candidate = lines[1]
        looks_like_title_repeat = candidate.strip().lower() == lines[0].strip().lower()
        looks_like_a_role = bool(_TITLE_KEYWORDS.search(candidate))
        # Sprint 15.8.3: a location line ("Amsterdam, Netherlands") is
        # just as plausible a line-2 candidate structurally as a real
        # company name — reuses _looks_like_location_line() (already
        # used elsewhere for parsing an actual location field) so this
        # fallback never mistakes one for the other.
        looks_like_a_location = _looks_like_location_line(candidate)
        if (
            not looks_like_title_repeat and not looks_like_a_role
            and not looks_like_a_location and not _is_noise_header(candidate)
        ):
            return candidate
    return "Unknown"


# ── Company-card structural signal (Sprint 15.8.3) ──────────────────────
#
# "<count> followers" / "<range> employees" is a structural shape many
# job-board company-profile cards share (LinkedIn, Indeed, Glassdoor all
# render a follower/employee count on a company snapshot) — a text
# pattern, not a LinkedIn CSS selector or DOM assumption. No existing
# tier in _guess_company() ever looked for this shape at all: the
# guarded line-2 fallback only inspects one specific line index right
# after the title, but a company card is a separate content block that
# doesn't reliably land there.
_COMPANY_CARD_FOLLOWERS = re.compile(r"(?i)^[\d,]+\+?\s+followers?\s*$")
_COMPANY_CARD_EMPLOYEES = re.compile(r"(?i)^[\d,]+(?:\s*[-–]\s*[\d,]+)?\+?\s+employees?\s*$")
# Sprint 15.8.4: a third company-card metric ("1,044 on LinkedIn" —
# how many of the company's employees are on LinkedIn), same structural
# shape as the two above.
_COMPANY_CARD_ON_LINKEDIN = re.compile(r"(?i)^[\d,]+\+?\s+on\s+linkedin\s*$")
_COMPANY_CARD_SCAN_WINDOW = 40  # this widget isn't always near the top, unlike title/company fields
_COMPANY_CARD_BACKTRACK = 4     # how far back from the anchor line to look for a name candidate
# "Follow"/"Following" — a UI action-button label that can sit between
# a company card's name and its follower count on some layouts; a
# generic UI-label exclusion (same spirit as "Easy Apply" elsewhere in
# this file), not a per-site selector.
_COMPANY_CARD_UI_LABELS = {"follow", "following"}


def _is_company_card_anchor(line: str) -> bool:
    return bool(
        _COMPANY_CARD_FOLLOWERS.match(line)
        or _COMPANY_CARD_EMPLOYEES.match(line)
        or _COMPANY_CARD_ON_LINKEDIN.match(line)
    )


def _plausible_company_name(candidate: str, title: str) -> bool:
    """The shared validation every structural company signal in this
    module applies to a heading-shaped candidate before trusting it as
    a name — the same checks the guarded line-2 fallback above already
    applies (title-repeat/role-keyword/noise-header), reused rather
    than re-implemented, plus two checks these signals specifically
    need: a location line must never be mistaken for a company name
    (Sprint 15.8.3's explicit "Bad: Company: 'Amsterdam'" case), and a
    bare UI action label ("Follow") isn't a name either.

    Sprint 15.8.5: the location/workplace-type check is only applied
    to a HEADING-shaped candidate (short, no terminal punctuation) —
    same fix as _guess_company_card_tagline()'s tagline validation
    (Sprint 15.8.4), applied here too. _CITY_COMMA's pattern
    (city+comma+more-text) can false-positive on an ordinary sentence
    under 60 characters with exactly one comma (e.g. "We are Super
    Technologies, simply known as Super." is 51 characters) — a real
    location/workplace value is always heading-shaped; a real company
    name candidate that happens to BE a full sentence should never be
    reached here in practice, but this guard makes the function safe
    regardless of what candidate a future caller passes it.

    Sprint 15.8.6: generalized from "_plausible_company_card_name" —
    originally written only for a company-card start-of-block
    candidate, this same validation applies unchanged to a candidate
    found via a completely different structural signal (a heading
    immediately before the resolved title, no card involved at all —
    see _guess_company_heading_before_title()). Renamed rather than
    duplicated: both signals answer the same question ("is this
    heading-shaped line plausibly a company name, given the resolved
    title"), just from different positional evidence. Also rejects an
    explicit "Company: X" labeled line here — not because the line is
    wrong, but because _guess_company_from_explicit_patterns() already
    parses that shape correctly (extracting just the value after the
    label); a positional signal accepting the whole unparsed line first
    would shadow that correct, more specific tier.
    """
    if not candidate or len(candidate) >= 60:
        return False
    if candidate.strip().lower() == title.strip().lower():
        return False
    if candidate.strip().lower() in _COMPANY_CARD_UI_LABELS:
        return False
    if _TITLE_KEYWORDS.search(candidate):
        return False
    if _COMPANY_PREFIX.match(candidate):
        return False
    if _is_noise_header(candidate):
        return False
    if _looks_like_heading(candidate) and (_looks_like_location_line(candidate) or _WORKPLACE_TYPE_LINE.match(candidate)):
        return False
    if _is_company_card_anchor(candidate):
        return False
    return True


class _CompanyCardBlock(NamedTuple):
    start_index: int  # first line of the contiguous card block — where the name is
    end_index: int     # first line AFTER the block — where a tagline would start


def _find_company_card_block(lines: list[str], anchor_index: int, title: str) -> _CompanyCardBlock:
    """Sprint 15.8.5: the ONE shared understanding of a heading-shaped
    block's extent, given an already-found anchor line — despite the
    name, `anchor_index` doesn't have to be a company-card anchor
    (`_is_company_card_anchor()` is only ever checked as one of several
    "still part of the block" conditions while walking, never required
    of the anchor itself). _guess_company_from_card() (needs the
    START, where the name is) and _guess_company_card_tagline() (needs
    the END, where a description would begin) both derive their answer
    from this function for a real company-card anchor; Sprint 15.8.7's
    _guess_company_heading_before_title() reuses the exact same walk
    with the resolved TITLE's own line as the anchor instead — same
    bounded backward tolerance, same stop conditions, one place that
    knows "how far back can a company name plausibly sit," rather than
    a second, independently-tuned tolerance for the no-card case.
    Originally written to fix a reported failure where company_overview
    found "Super" via one scan while company resolution failed via
    another — the general principle (one shared block-boundary
    understanding, not several disagreeing scans) is what generalizes
    here.

    A line is still "part of the card" if it's another anchor
    (followers/employees/on LinkedIn), a UI action label ("Follow"), or
    a short heading-shaped line (an industry name, e.g.) — walked in
    BOTH directions from the anchor, bounded by
    _COMPANY_CARD_BACKTRACK/_COMPANY_CARD_SCAN_WINDOW so this can never
    run away across unrelated content.

    The backward walk additionally, explicitly stops at the job title
    itself — never treats it as part of the card, even though a short
    title can otherwise look identical to a heading-shaped card field.
    This is what makes the START the company's actual NAME (the first
    line of the card) rather than merely "whichever card-shaped line
    happens to be nearest the anchor" (the previous behavior, which
    could return an industry name sitting between the real name and
    the anchor — the name is always first in a real card, so walking
    all the way back to where the block starts, not stopping at the
    first valid-looking line, is the fix).

    A second, equally important stop condition: a location line
    ("Amsterdam Area") or a bare workplace-type line ("Hybrid") is
    NEVER part of a company card, even though both are heading-shaped
    like a real card field — without this check, the backward walk
    would otherwise keep going straight through a page's own location/
    workplace-mode lines (which often sit just above the card,
    unrelated to it) and misidentify one of them as the block's start.
    Reuses _looks_like_location_line()/_WORKPLACE_TYPE_LINE — the same
    checks _plausible_company_name() already applies to a
    candidate — as an explicit boundary here instead."""
    start = anchor_index
    earliest = max(0, anchor_index - _COMPANY_CARD_BACKTRACK)
    for index in range(anchor_index - 1, earliest - 1, -1):
        line = lines[index]
        if _line_matches_title(line, title):
            break  # the job title itself — never part of the card
        if _looks_like_heading(line) and (_looks_like_location_line(line) or _WORKPLACE_TYPE_LINE.match(line)):
            break  # a location/workplace-type line is never part of a company card
        if not (
            _is_company_card_anchor(line)
            or line.strip().lower() in _COMPANY_CARD_UI_LABELS
            or _looks_like_heading(line)
        ):
            break
        start = index

    end = anchor_index + 1
    limit = min(len(lines), anchor_index + 1 + _COMPANY_CARD_SCAN_WINDOW)
    while end < limit:
        line = lines[end]
        if _is_any_section_header(line):
            break
        if line.strip().lower() in _COMPANY_CARD_UI_LABELS or _is_company_card_anchor(line):
            end += 1
            continue
        if _looks_like_heading(line):
            end += 1
            continue
        break

    return _CompanyCardBlock(start_index=start, end_index=end)


def _guess_company_from_card(lines: list[str], title: str) -> Optional[str]:
    """Scans for the first "<count> followers"/"<range> employees" line
    (the company-card anchor) within the first _COMPANY_CARD_SCAN_WINDOW
    lines, then uses _find_company_card_block() to locate the START of
    that card — the company's actual name is always the first line of
    the block, not merely the nearest card-shaped line to the anchor
    (see that function's own docstring for the bug this fixes). If the
    block's start line doesn't pass _plausible_company_name(),
    tries the next anchor found further down rather than giving up on
    the first one.

    Sprint 15.8.5: `title` is the caller's ALREADY-RESOLVED job title,
    not a re-guess. This function used to approximate it as `lines[0]`
    — a reasonable guess for most postings, but wrong exactly when the
    extracted text leads directly into the company card with no
    separate title line first (lines[0] IS the company name in that
    case). That mismatch made _find_company_card_block()'s backward
    walk stop one line too early — mistaking the real company name for
    "the title" and refusing to walk past it — reproduced directly and
    fixed by threading the real title through instead of re-deriving a
    wrong one here."""
    if not lines:
        return None
    for index, line in enumerate(lines[:_COMPANY_CARD_SCAN_WINDOW]):
        if not _is_company_card_anchor(line):
            continue
        block = _find_company_card_block(lines, index, title)
        candidate = lines[block.start_index]
        if _plausible_company_name(candidate, title):
            return candidate
    return None


def _guess_company_heading_before_title(lines: list[str], title: str) -> Optional[str]:
    """Sprint 15.8.6: the mirror of _immediately_precedes_company_card()
    — that function asks "is the line AFTER this title candidate a
    company-card anchor" (used to stop title-guessing from grabbing a
    company's own name); this asks "is the line BEFORE the already-
    resolved title a plausible company name" (used to find the company
    when there's no card at all).

    Root cause this fixes: a real reported LinkedIn shape has no
    follower/employee/industry lines anywhere — just a plain vertical
    header, company name then title then location then workplace type
    ("Super" / "Staff Machine Learning Engineer" / "Amsterdam Area..."
    / "Hybrid"). _guess_company_from_card() correctly finds no anchor
    and contributes nothing; the weak line-2 fallback (_guess_company())
    is hardcoded to only ever consider lines[1] as a candidate — it has
    no path to lines[0], because it assumes the OPPOSITE order (title
    first, company second). Neither is a company-card problem, and
    neither generalizes to "the company can precede the title" — this
    function is that missing generalization, expressed the same way
    every other structural signal in this cascade is: as a position
    relative to an already-known anchor (here, the resolved title,
    found by its own text rather than assumed to be lines[0] or
    lines[1]), validated by the exact same _plausible_company_name()
    check the company-card signal already uses — no new pattern.

    Sprint 15.8.7: uses the SAME bounded backward walk
    _find_company_card_block() already performs for a company-card
    anchor, called here with the title's own line as the anchor point
    instead — see that function's own docstring for why this reuse is
    valid (it never actually required a card-shaped anchor). Fixes a
    real gap in the original version of this function, which only
    ever checked the single immediately-preceding line: reproducing
    the exact reported shape in isolation passed, because that
    reproduction happened to have zero lines between the company name
    and the title — but the underlying assumption (zero tolerance for
    anything in between) was never actually tested against a layout
    with even one intervening line (a badge, a rating widget, any
    other heading-shaped noise a real browser extraction can insert),
    which the company-card walk was already built to tolerate.
    Sharing the walk means both "company before a card anchor" and
    "company before the title, no card at all" have one tolerance,
    not two independently-tuned ones that can silently disagree.

    Finds the first line (within _HEADER_WINDOW) matching `title`
    verbatim, then walks backward from it exactly as
    _find_company_card_block() would from a card anchor; the walk's
    own stop conditions (the title itself, a location/workplace-type
    line) keep it from running past unrelated content. Silent None
    otherwise (title not found in this window, title is the first
    line, the walk finds nothing before it, or the resulting candidate
    doesn't validate) — same "no signal, no guess" contract as every
    sibling tier."""
    if not lines or not title:
        return None
    for index, line in enumerate(lines[:_HEADER_WINDOW]):
        if not _line_matches_title(line, title):
            continue
        if index == 0:
            return None
        block = _find_company_card_block(lines, index, title)
        if block.start_index == index:
            return None  # the walk found nothing usable before the title
        candidate = lines[block.start_index]
        return candidate if _plausible_company_name(candidate, title) else None
    return None


def _guess_company_from_structure(lines: list[str], title: str) -> Optional[str]:
    """The ONE structural/positional tier _resolve_company() calls —
    company-card detection and the heading-before-title signal (Sprint
    15.8.6) grouped together here rather than as two separate branches
    in the cascade, since both answer the same underlying question
    ("does the page's own layout tell us where the company name is,"
    as opposed to matching company-identifying words in a sentence)
    and both defer to the same validator (_plausible_company_name()).
    Card detection tried first — a multi-line-corroborated signal
    (name plus follower/employee counts) is stronger evidence than a
    single adjacent heading; the heading-before-title signal only ever
    contributes when no card exists at all, which is exactly the real
    case that motivated adding it.

    Sprint 15.8.8: both signals locate a company by finding WHERE the
    title sits in `lines` via an exact (case-insensitive) text match —
    they need a line that's guaranteed to literally appear in the raw
    text, not necessarily the same string the caller will ultimately
    display as the job's title. For RuleBasedExtractor these are
    almost always identical; for LLMExtractor they can genuinely
    differ, since the model's own `title` field is a semantic
    paraphrase (e.g. "Senior ML Engineer" for a raw line that reads
    "Senior Machine Learning Engineer") — reproduced directly: with a
    paraphrased title, both signals silently return None even though
    the company sits exactly where they're designed to find it, while
    _derive_company_overview() (which has no dependency on the title
    string at all) succeeds on the identical text. That divergence —
    not a company-card gap, not a validation-rejection gap — is what
    produces "company_overview finds it, company resolution doesn't"
    whenever the caller's title isn't verbatim.

    Fixed with a second attempt using a separately-derived title that
    IS guaranteed to be a literal line: the same deterministic
    _guess_title_with_confidence() RuleBasedExtractor already trusts
    when no smarter signal exists — not a new heuristic, reused
    as-is. Only runs if the caller's own title didn't already work
    (skipped entirely, zero behavior change, when it did) and only if
    the two differ (no point retrying an identical value)."""
    from_card = _guess_company_from_card(lines, title) or _guess_company_heading_before_title(lines, title)
    if from_card:
        return from_card
    structural_title, _ = _guess_title_with_confidence(lines)
    if structural_title.strip().lower() == title.strip().lower():
        return None
    return _guess_company_from_card(lines, structural_title) or _guess_company_heading_before_title(lines, structural_title)


def _skip_leading_company_card(lines: list[str], title: str) -> int:
    """Sprint 15.8.5: when the extracted text leads directly into the
    company card (no title line before it — the exact "company card
    appears before the actual job title" layout this sprint's report
    describes), _HEADER_WINDOW's fixed-size scan for location/
    workplace-type starting at line 0 only sees card content (name,
    industry, follower/employee counts, tagline) — pushing the real
    header fields ("Amsterdam Area", "Hybrid") past the window entirely
    even though they're only a few lines further down. Reproduced
    directly: a card of realistic size (name + industry + 3 anchor
    lines + a tagline sentence) already exhausts _HEADER_WINDOW on its
    own. Returns the card's end_index (where scanning should actually
    start) when one leads the text, else 0 — reuses the same
    _find_company_card_block() both company resolution and the
    tagline capture already rely on, no new pattern."""
    if not lines:
        return 0
    for index, line in enumerate(lines[:_COMPANY_CARD_SCAN_WINDOW]):
        if not _is_company_card_anchor(line):
            continue
        block = _find_company_card_block(lines, index, title)
        return block.end_index if block.start_index == 0 else 0
    return 0


def _resolve_company(
    lines: list[str], posting: Optional[dict], meta: dict[str, str], title: str, *,
    weak_fallback: bool = True,
) -> Optional[str]:
    """Sprint 15.8.3: the single shared company-resolution cascade —
    both RuleBasedExtractor entry points (extract_from_text()/
    extract_from_url()) AND LLMExtractor (which overlays this
    deterministic result over its own free-text inference — see
    llm.py's _extract()) go through this one function, so there is
    exactly one place that knows the priority order:

        JSON-LD hiringOrganization
          > structured og:site_name metadata
          > structural/positional signals (_guess_company_from_structure():
            company-card detection, then — Sprint 15.8.6 — a heading
            immediately before the resolved title when no card exists)
          > existing CONFIDENT text heuristics (explicit "Company:"
            field / sentence patterns — _guess_company_from_explicit_patterns())
          > [weak_fallback only] the blind "assume line 2" last resort
          > None (no deterministic answer at all — the caller decides
            what happens next: "Unknown" for rule-based, the model's
            own guess for the LLM backend)

    `title` (Sprint 15.8.5) is the caller's own already-resolved job
    title, forwarded to _guess_company_from_structure() so both of its
    signals reason about the real title rather than approximating it —
    see _guess_company_from_card()'s and
    _guess_company_heading_before_title()'s own docstrings.

    `weak_fallback` (default True, RuleBasedExtractor's behavior,
    unchanged) controls whether the last, no-real-signal "assume line
    2" guess participates. LLMExtractor passes `weak_fallback=False`:
    that guess has no positive signal behind it at all (it exists only
    because RuleBasedExtractor has nothing better to fall back to), and
    letting it override a capable model's own contextual inference would
    make LLM-backend extraction quality *worse*, not better — a blind
    guess must never outrank real reasoning. Every tier ABOVE it (JSON-
    LD, meta, the structural signals, and the confident explicit
    patterns) stays available to both backends regardless, since those
    are all real, positive signals.

    Purely an orchestrator — no new extraction logic of its own, never
    raises, never guesses beyond what each tier already guarantees."""
    if posting:
        from_jsonld = _company_from_jsonld(posting)
        if from_jsonld != "Unknown":
            return from_jsonld
    site_name = meta.get("og:site_name") if meta else None
    if site_name:
        return site_name
    from_structure = _guess_company_from_structure(lines, title)
    if from_structure:
        return from_structure
    if weak_fallback:
        from_heuristics = _guess_company(lines)
        return from_heuristics if from_heuristics != "Unknown" else None
    return _guess_company_from_explicit_patterns(lines)


_COMPOUND_METADATA_SEPARATORS = re.compile(r"[·|]")


def _reliable_location_value(line: str) -> Optional[str]:
    """Sprint 15.8.4: the actual fix for "location = Hybrid · 2 weeks
    ago · 87 applicants" — a delimiter-separated compound metadata line
    (LinkedIn and similar sites render "Hybrid · 2 weeks ago · 87
    applicants" as one line) is not a clean location value even when it
    contains a remote/hybrid keyword _LOCATION_HINT's substring search
    would otherwise match. The precise, full-line shapes (city+comma,
    city/region/country, "<City> Area") are trusted regardless — they
    can't accidentally match a compound line by construction (the
    pattern requires the WHOLE line, delimiters included, to fit the
    shape). Only the loose _LOCATION_HINT substring match gets the
    extra compound-line guard, since it's the one imprecise check.

    Sprint 15.8.5: _CITY_COMMA/_CITY_REGION_COUNTRY are only trusted
    for a HEADING-shaped line (short, no terminal punctuation) — the
    same false-positive fix applied to _guess_company_card_tagline()
    (Sprint 15.8.4) and _plausible_company_name() above.
    _CITY_COMMA's pattern doesn't actually require anything city-
    shaped, just "capitalized text, one comma, then letters/spaces to
    the end" — an ordinary sentence within _guess_location()'s scan
    window (the first _HEADER_WINDOW lines, which a company card can
    sit close to) could otherwise be wrongly returned as the entire
    location value. _AREA_LOCATION isn't gated the same way — its
    pattern requires ending in the literal word "Area", precise enough
    that an ordinary sentence coincidentally ending that way is not a
    realistic risk.

    Sprint 15.8.6: renamed from "_is_reliable_location_candidate" (a
    bool) to reflect what it now returns — the clean value, not just
    whether the line was acceptable. A real reported shape ("Amsterdam
    Area · Reposted 2 days ago") has a genuinely valid _AREA_LOCATION
    value as its LEADING segment, but the old compound-separator check
    rejected the entire line outright the moment any "·"/"|" appeared,
    discarding a real location along with the metadata trailing it —
    the same function, two opposite failure directions (Sprint 15.8.4:
    don't let junk-with-a-keyword become the location; this sprint:
    don't let junk-after-a-real-value destroy the location). Fixed by
    normalizing FIRST: validate the segment before the first separator
    against the same precise shapes as before, and — critically — a
    compound line's segment is NEVER allowed to fall through to the
    loose _LOCATION_HINT keyword search, only a genuinely separator-
    free line still gets that (preserves the original Sprint 15.8.4
    fix exactly: "Hybrid · 2 weeks ago · 87 applicants" still doesn't
    validate, since its segment "Hybrid" matches none of the precise
    shapes and the loose fallback is walled off for any compound
    line)."""
    segment = _COMPOUND_METADATA_SEPARATORS.split(line, maxsplit=1)[0].strip()
    has_separator = segment != line.strip()
    if _AREA_LOCATION.match(segment):
        return segment
    if _looks_like_heading(segment) and (_CITY_COMMA.match(segment) or _CITY_REGION_COUNTRY.match(segment)):
        return segment
    if has_separator:
        return None
    return line if _LOCATION_HINT.search(line) else None


def _guess_location(lines: list[str], title: str) -> str:
    # Sprint 15.8.5: scan starting after a leading company card, if one
    # is present — see _skip_leading_company_card()'s docstring.
    offset = _skip_leading_company_card(lines, title)
    window = lines[offset:offset + _HEADER_WINDOW]
    for line in window:
        m = _LOCATION_PREFIX.match(line)
        if m:
            return m.group(1).strip()
    for line in window:
        # Sprint 15.8.4: a bare "Hybrid"/"Remote"/"On-site" line is a
        # workplace-type value, not a location — _guess_workplace_type_
        # from_text() (below) is the dedicated source for that, so it's
        # excluded here rather than accidentally returned as location.
        if _WORKPLACE_TYPE_LINE.match(line):
            continue
        value = _reliable_location_value(line)
        if value:
            return value
    return "Unknown"


def _guess_workplace_type_from_text(lines: list[str], title: str) -> Optional[str]:
    """Sprint 15.8.4: a text-heuristic source for workplace_type — a
    standalone "Remote"/"Hybrid"/"On-site" line, distinct from a real
    geographic location. Sprint 15.8.1 deliberately left this JSON-LD-
    only ("no standard 'Hybrid' value in schema.org, never guessed from
    ambiguous prose"); this isn't a guess from ambiguous prose, it's a
    literal, labeled, standalone value in the page's own text — a much
    more reliable signal, and this sprint's explicit ask. Callers must
    only use this to fill in a MISSING workplace_type — JSON-LD stays
    Priority 1 and is never overridden by this (see extract_from_text()/
    extract_from_url() below).

    Sprint 15.8.5: scan starting after a leading company card, if one
    is present — same window-alignment fix as _guess_location()."""
    offset = _skip_leading_company_card(lines, title)
    for line in lines[offset:offset + _HEADER_WINDOW]:
        m = _WORKPLACE_TYPE_LINE.match(line)
        if m:
            value = m.group(1).lower().replace("-", "").replace(" ", "")
            if value == "remote":
                return "Remote"
            if value == "hybrid":
                return "Hybrid"
            if value in ("onsite", "inoffice"):
                return "On-site"
    return None


def _job_id_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _identity_signature(title: str, company: str, text: str) -> str:
    """
    What identifies "the same job" for pasted/file input.

    Hashing the full raw text (the old behavior) made identity unstable:
    re-pasting "the same" posting rarely produces byte-identical text —
    stray whitespace, an extra blank line, or a slightly different manual
    selection from the page all change the hash, so the same job silently
    got tracked twice. Title+company is far more stable across re-pastes
    and is exactly what a human means by "the same job posting".

    Falls back to the full text only when neither was confidently
    extracted (both still at their "unknown" default) — otherwise two
    unrelated, unparseable postings would collide onto one identity.
    """
    if title != "Untitled" or company != "Unknown":
        return f"{title.strip().lower()}|{company.strip().lower()}"
    return text


_USER_AGENT = (
    "Mozilla/5.0 (compatible; job-tracker/1.0; +https://github.com/)"
)

# LinkedIn (and similar sites) serve a 200 OK login/auth-wall page instead of
# an HTTP error when a job posting isn't publicly viewable. Status codes
# alone can't catch that, so we also scan the body for tells.
_BLOCKED_MARKERS = (
    "join linkedin",
    "authwall",
    "sign in to view",
    "verify you are human",
    "unusual activity",
)


def _looks_blocked(html: str) -> bool:
    lowered = html.lower()
    return any(marker in lowered for marker in _BLOCKED_MARKERS)


def fetch_job_page(url: str, timeout: float = 10.0) -> str:
    """
    Fetch a job posting page over HTTP(S) using only the standard library.

    Never lets a raw network exception escape — everything collapses into
    JobFetchError with a `reason` of "login required", "blocked request", or
    "unavailable page" so the CLI can show a friendly fallback instead of a
    stack trace.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise JobFetchError(url, "unavailable page", "not a valid http(s) URL")

    request = Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        if exc.code in (401, 403, 429, 999):  # 999 is LinkedIn's anti-scraping code
            raise JobFetchError(url, "blocked request", f"HTTP {exc.code}") from exc
        raise JobFetchError(url, "unavailable page", f"HTTP {exc.code}") from exc
    except URLError as exc:
        raise JobFetchError(url, "unavailable page", str(exc.reason)) from exc
    except OSError as exc:
        raise JobFetchError(url, "unavailable page", str(exc)) from exc

    if _looks_blocked(html):
        raise JobFetchError(url, "login required", "page requires sign-in")

    return html


class _PageParser(HTMLParser):
    """Pulls out <title>, <meta property/name=...content=...>, JSON-LD
    <script> blocks, and plain visible text — all in one pass, stdlib only.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title = ""
        self.jsonld_blocks: list[str] = []
        self._text_parts: list[str] = []
        self._in_title = False
        self._in_jsonld = False
        self._skip_text = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attrs_d = dict(attrs)
        if tag == "meta":
            key = (attrs_d.get("property") or attrs_d.get("name") or "").lower()
            content = attrs_d.get("content")
            if key and content is not None:
                self.meta[key] = content
        elif tag == "title":
            self._in_title = True
        elif tag == "script":
            self._skip_text = True
            self._in_jsonld = (attrs_d.get("type") or "").lower() == "application/ld+json"
        elif tag == "style":
            self._skip_text = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "script":
            self._skip_text = False
            self._in_jsonld = False
        elif tag == "style":
            self._skip_text = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._in_jsonld:
            self.jsonld_blocks.append(data)
        elif not self._skip_text:
            stripped = data.strip()
            if stripped:
                self._text_parts.append(stripped)

    @property
    def text(self) -> str:
        return "\n".join(self._text_parts)


def _strip_html(fragment: str) -> str:
    """Reduce an HTML fragment (e.g. a JSON-LD `description` field) to text."""
    parser = _PageParser()
    parser.feed(fragment)
    return parser.text


def _find_jobposting(data: Any) -> Optional[dict]:
    """Locate a schema.org JobPosting node in parsed JSON-LD, handling both
    bare objects/lists and `@graph`-wrapped documents."""
    nodes = data if isinstance(data, list) else [data]
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("@type") == "JobPosting":
            return node
        graph = node.get("@graph")
        if isinstance(graph, list):
            for sub in graph:
                if isinstance(sub, dict) and sub.get("@type") == "JobPosting":
                    return sub
    return None


def _location_from_jsonld(posting: dict) -> str:
    location = posting.get("jobLocation")
    if isinstance(location, list):
        location = location[0] if location else None
    if isinstance(location, dict):
        address = location.get("address")
        if isinstance(address, dict):
            parts = [
                address.get("addressLocality"),
                address.get("addressRegion"),
                address.get("addressCountry"),
            ]
            parts = [p for p in parts if p]
            if parts:
                return ", ".join(parts)
    if posting.get("jobLocationType") == "TELECOMMUTE":
        return "Remote"
    return "Unknown"


def _company_from_jsonld(posting: dict) -> str:
    org = posting.get("hiringOrganization")
    if isinstance(org, dict):
        name = org.get("name")
        if name:
            return str(name)
    return "Unknown"


def _department_from_jsonld(posting: dict) -> Optional[str]:
    """schema.org JobPosting has no standard "department" property, but
    some employers include one anyway (e.g. under `employmentUnit` or a
    plain `department` key) — use it opportunistically when present."""
    for key in ("department", "employmentUnit"):
        value = posting.get(key)
        if isinstance(value, dict):
            value = value.get("name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


# ── Structured metadata preservation (Sprint 15.8.1) ────────────────────
#
# Four more standard schema.org JobPosting properties, read the exact
# same way _company_from_jsonld()/_location_from_jsonld()/
# _department_from_jsonld() already are — JSON-LD only, no text-fallback
# heuristic for any of them (unlike title/company/location, which
# already had one from before this sprint). A wrong guess here has no
# correction mechanism the way title's confidence tiers do, so Priority
# 3 (plain text, no structured source) leaves all four at their default
# (None) rather than inventing a heuristic — consistent with this
# sprint's explicit "no LinkedIn-specific/brittle selectors" constraint.

def _employment_type_from_jsonld(posting: dict) -> Optional[str]:
    value = posting.get("employmentType")
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _posted_at_from_jsonld(posting: dict) -> Optional[datetime]:
    """schema.org's `datePosted` is typically ISO 8601 (what Google's
    own structured-data guidance requires) — datetime.fromisoformat()
    handles that directly. Some employers get this wrong (a non-ISO
    date string, or omit it); malformed/missing values degrade to None,
    never a crash — the same "not extracted, never an error" contract
    every other optional field on ExtractionResult already has."""
    value = posting.get("datePosted")
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _workplace_type_from_jsonld(posting: dict) -> Optional[str]:
    """"Remote" only, when jobLocationType is schema.org's one standard
    remote-work value (TELECOMMUTE) — there is no standard "Hybrid"
    value in the spec, so Hybrid vs. on-site is never guessed here; a
    generic, non-site-specific reader has no reliable way to tell them
    apart. None (field simply absent downstream) is the honest
    answer for every posting that isn't explicitly remote."""
    if posting.get("jobLocationType") == "TELECOMMUTE":
        return "Remote"
    return None


def _safe_http_url(value: Any) -> Optional[str]:
    """Only ever returns a plain http(s) URL, or None — the one piece
    of JSON-LD-sourced data a downstream consumer may render as a clickable
    link (`company_url`), so accepting an arbitrary scheme here
    (`javascript:`, `data:`, ...) would be a real stored-XSS-adjacent
    risk the moment it's rendered as an <a href>. Never raises on
    malformed input — degrades to None like every other optional field."""
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return candidate


def _company_url_from_jsonld(posting: dict) -> Optional[str]:
    org = posting.get("hiringOrganization")
    if not isinstance(org, dict):
        return None
    for key in ("url", "sameAs"):
        value = org.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        safe = _safe_http_url(value)
        if safe:
            return safe
    return None


def _parse_structured_hints(structured_hints: Optional[dict]) -> tuple[Optional[dict], dict[str, str]]:
    """Interprets the raw `{"json_ld": [...], "meta": {...}}` a browser
    client forwarded (Sprint 15.7 — see extension/content.js and
    browser/bookmarklet.js) into a schema.org JobPosting node (if one's
    present) plus a plain meta-tag dict. Reuses _find_jobposting() — the
    exact same JSON-LD interpretation extract_from_url() already does
    for a server-side fetch — rather than a second parser, so there is
    only ever one place that knows how to read a JobPosting node.

    Never raises: a missing/malformed/JSON-LD-free block just means "no
    structured data available," exactly like a plain-text paste (no
    hints at all) or a page with no JSON-LD already behave."""
    if not structured_hints or not isinstance(structured_hints, dict):
        return None, {}

    posting: Optional[dict] = None
    for block in structured_hints.get("json_ld") or []:
        if not isinstance(block, str) or not block.strip():
            continue
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        posting = _find_jobposting(data)
        if posting:
            break

    meta = structured_hints.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    return posting, meta


# ── Job Understanding from plain description text (Sprint 15.7) ────────────
#
# Populates the same summary/responsibilities/requirements/
# required_skills/preferred_skills fields llm.py already produces from
# a model — a lower-quality, zero-cost approximation for the
# rule-based path, not a replacement for the LLM path. Generic
# section-header vocabulary only, same
# "structural pattern, not per-site markup" rule as
# _guess_structured_header()/_NOISE_HEADERS above — no LinkedIn-
# specific selectors.

_SECTION_HEADERS: dict[str, re.Pattern] = {
    # Sprint 15.8.2 additions (named, bounded — not an open-ended list):
    # "your responsibilities"/"your core responsibilities" (a "your"-
    # prefixed variant of the existing bare "responsibilities"), "what
    # you will do" spelled out (the existing "what you.?ll do" only
    # matched the contraction — ".?" allows at most one character, so
    # "you will do" never matched it), and "the role" (already a
    # _NOISE_HEADERS boundary marker, now also a responsibilities
    # trigger — a real, reported posting shape, not a guess).
    #
    # Sprint 18 Round 2 (Responsibilities Semantic Audit → fix): "what
    # the role involves" — a real, corpus-evidenced header (a real anonymized production case, see docs/regression-corpus.md)
    # this list didn't cover, causing a genuine, well-formed 6-item duty
    # list to be silently dropped entirely (see
    # docs/regression-corpus.md for the full history).
    # Added TOGETHER with requirements' "what we are looking for" below
    # in the same change — the audit confirmed widening this list alone,
    # without also widening requirements', would make
    # _collect_section_items() run straight through the next (still-
    # unrecognized) requirements header as if it were more responsibility
    # content, mislabeling real requirement bullets as responsibilities.
    # Sprint 21 Round 4: "this is an opportunity to" — a real,
    # corpus-evidenced header (a real anonymized production case — see
    # docs/regression-corpus.md) this
    # list didn't cover, silently dropping a clean, well-formed 10-item
    # duty list. Added TOGETHER with requirements' "what you will bring"
    # below, in the same change — confirmed via a non-destructive
    # in-memory hypothesis test (same methodology as Sprint 18 Round 2)
    # that widening this list alone lets collection run straight through
    # the still-unrecognized "What You Will Bring" line and into the
    # first requirement bullet before the item cap stops it.
    "responsibilities": re.compile(
        r"(?i)^(responsibilities|key responsibilities|your responsibilities|"
        r"your core responsibilities|what you.?ll do|what you will do|"
        r"duties|role overview|day.to.day|what you.?ll be doing|the role|"
        r"what the role involves|this is an opportunity to)\s*:?\s*$"
    ),
    # Sprint 18 Round 2: "what we are looking for" — the existing
    # "what we.?re looking for" only matches the contraction ("we're";
    # `.?` already covers straight or curly apostrophe, one character
    # either way) — the spelled-out "we are" needs its own alternative,
    # the same "contraction + spelled-out" pairing already used for
    # responsibilities' "what you.?ll do|what you will do" above. Added
    # in the same change as responsibilities' "what the role involves"
    # — see that entry's comment for why the two are coupled.
    # Sprint 21 Round 4: "what you will bring" — the existing "what you
    # bring" only matches that exact phrase, not the spelled-out "will"
    # variant this real posting (a real anonymized production case, see docs/regression-corpus.md) actually uses. Added only
    # because the responsibilities coupling above requires it (verified,
    # not assumed) — no other requirements-side change made this round.
    "requirements": re.compile(
        r"(?i)^(requirements|qualifications|minimum qualifications|"
        r"basic qualifications|what you.?ll need|who you are|"
        r"skills (and|&) experience|your skills and experience|"
        r"what we.?re looking for|what we are looking for|"
        r"what you bring|what you will bring)\s*:?\s*$"
    ),
    "preferred": re.compile(
        r"(?i)^(preferred qualifications|nice to have|nice.to.haves|"
        r"bonus points|preferred skills|good to have|preferred|"
        r"it.?s a plus if)\s*:?\s*$"
    ),
    # Sprint 15.7: a distinct "Education:" section — kept separate from
    # (not merged into) "requirements" so it can be captured into its
    # own education_requirements field; the same bullet still ends up
    # in `requirements` too via _extract_understanding() below, since a
    # degree requirement is still a requirement — this is additive, not
    # a re-categorization.
    "education": re.compile(
        r"(?i)^(education|education requirements|minimum education|"
        r"academic requirements)\s*:?\s*$"
    ),
}

# Sprint 15.7: the company-related subset of _NOISE_HEADERS — reused
# here (not duplicated) to find where a posting's own "About Us"/"Who
# We Are" text starts, so its content can be captured into
# company_overview. _NOISE_HEADERS/_is_noise_header()'s original job
# (never treating this line as a title, and stopping _derive_summary()
# before it) is completely unchanged; this only reads the same set for
# a second purpose.
_COMPANY_OVERVIEW_HEADERS = {"about us", "about the company", "company overview", "who we are"}

_MAX_SECTION_ITEMS = 12
_MAX_SECTION_ITEM_LENGTH = 200
_BULLET_PREFIX = re.compile(r"^[\-\*•●▪●▪]\s*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MIN_SUMMARY_SENTENCE_WORDS = 6

# Sprint 15.7/15.8.2: generic UI-chrome sentence patterns — the actual
# root cause of "summary/company_overview contains raw LinkedIn UI
# text" in production. extract_from_text()'s understanding_source
# falls back to a browser client's raw extracted text (extension/
# content.js's <main> innerText) whenever JSON-LD is thin/absent, and
# that text includes interactive page chrome, not just the posting
# body — sentences like "See who Acme Corp has hired for this role" or
# "Over 100 people clicked apply" are grammatically real sentences
# (>=6 words, no obviously-heading shape) that a plain length filter
# alone can't catch. Matched by structural phrasing any job board's
# chrome tends to share, same "pattern, not per-site selector"
# convention _NOISE_HEADER_PATTERNS already uses — not LinkedIn-
# specific text.
#
# Sprint 15.8.2 grouped/broadened this after a real reported example
# (an IMC ML Engineer posting) showed the Sprint 15.7 list was too
# narrow to generalize: it covered "\d+\s+applicants?" but not "N
# people clicked apply"; it had no pattern at all for profile-match
# prompts ("Your profile...", "See how you compare..."), alert prompts
# ("Set alert for similar jobs"), data-attribution lines ("Based on
# LinkedIn data"), or promotion/freshness labels ("Promoted by hirer",
# "Reposted"). Grouped by noise *category* below so a future gap is
# easier to place correctly rather than appended as one more ungrouped
# alternative — still a fixed, reviewed list, not an open-ended one;
# see docs/regression-corpus.md for why regression
# tests for this pattern must be built from real reported examples.
_UI_CHROME_SENTENCE = re.compile(
    r"(?i)\b("
    # apply-button / applicant-activity chrome
    r"see who|easy apply|save this job|report this job|"
    r"\d+\s+applicants?|\d+\s+people\s+clicked\s+apply|clicked\s+apply|"
    # profile-matching / comparison prompts
    r"your\s+profile|see\s+how\s+you\s+compare|"
    # notification / alert prompts
    r"get\s+notified|set\s+alert|similar\s+jobs|people\s+also\s+viewed|"
    # data-attribution lines
    r"based\s+on\s+(?:linkedin|glassdoor|indeed)?\s*data|"
    # promotion / freshness labels
    r"promoted\s+by\s+hirer|reposted|"
    # generic page controls
    r"show\s+more|show\s+less|sign\s+in\s+to|"
    # Sprint 15.8.4: company-card metrics — the same "<count> followers"/
    # "<count> employees" shape _guess_company_from_card() already
    # recognizes for company-NAME detection (Sprint 15.8.3), now also
    # wired into this shared filter so summary/company_overview/LLM
    # prompt/LLM validation (every one of this pattern's consumers) all
    # correctly treat these as noise too — one shared pattern, not a
    # second purpose-specific list.
    r"\d[\d,]*\+?\s+followers?|\d[\d,]*\+?\s+employees?|\d[\d,]*\+?\s+on\s+linkedin|"
    r"\bfollow\b|"
    # recruiter/follow-prompt widgets
    r"interested\s+in\s+working|share\s+that\s+they.?re\s+interested"
    r")\b"
)


# Sprint 15.8.8: a "read more"/"see more" TRUNCATION LINK label, not a
# sentence to reject outright — a real reported example
# (company_overview containing literal "... more"/"… more" fragments)
# traced to a LinkedIn company-page UI affordance: text truncated in
# the rendered page with a clickable "…more" link, captured verbatim
# by innerText extraction. The FULL underlying text isn't present in
# the extracted payload at all (the DOM never rendered it — expanding
# the link is a client-side action browser extraction doesn't
# perform), so there is nothing to "recover"; the only correct fix is
# removing the literal control-label substring while preserving
# whatever real (if truncated) prose surrounds it. Deliberately a
# SUBSTRING strip, not a _UI_CHROME_SENTENCE whole-line/whole-sentence
# reject: unlike "show more"/"show less" (a control that's usually its
# own standalone line), "…more" is regularly glued directly onto the
# end of otherwise-real truncated content on the same line — rejecting
# the whole line would throw away genuine (if incomplete) prose the
# user should still see. Matches both the Unicode ellipsis (…) and the
# ASCII "..." spelling, with or without a space before "more".
_TRUNCATION_MARKER = re.compile(r"(?i)(?:…|\.\.\.)\s*more\b")


def _strip_truncation_markers(text: str) -> str:
    """Removes just the "more" control-label, normalizing whatever
    ellipsis preceded it to a literal "..." — NOT a full removal.
    Discovered via a real end-to-end regression while adding this fix:
    stripping "...more"/"…more" down to nothing also strips the only
    terminal punctuation the truncated sentence had, which then makes
    _looks_like_heading() (used right after this by
    _derive_company_overview() to filter out sub-headers, see that
    function's own docstring) mistake genuine truncated prose for a
    heading and drop it too — the two Sprint 15.8.8 fixes fighting
    each other. Keeping a literal "..." preserves the "this is a real,
    if incomplete, sentence" signal _looks_like_heading() already
    depends on, regardless of whether the source used the Unicode
    ellipsis or three ASCII periods."""
    return _TRUNCATION_MARKER.sub("...", text).strip()


def _strip_ui_noise_lines(lines: list[str]) -> list[str]:
    """Drops any line matching _UI_CHROME_SENTENCE — the one shared
    noise filter both _derive_summary()/_derive_company_overview()
    (below) and extraction/llm.py's prompt-building reuse, so there is
    only ever one place that knows what "chrome" looks like. This is
    part of this module's role as the *safety layer* in the hybrid
    extraction architecture (deterministic noise filtering + section-
    boundary hints + validation) — the LLM backend is what actually
    does summarization/understanding when enabled; this function's job
    is only to keep obvious page chrome out of whatever text a
    downstream consumer (rule-based derivation, or an LLM prompt) sees,
    not to understand or rewrite anything itself.

    Sprint 15.8.8: also strips a "…more"/"...more" truncation-link
    label from each line before the chrome check — see
    _strip_truncation_markers()'s own docstring. That helper
    deliberately normalizes the marker to a literal "..." rather than
    removing it outright (see its own docstring for why — stripping to
    nothing broke _looks_like_heading()'s truncated-prose signal for
    the GLUED case), so a line with real content keeps that content,
    ellipsis and all.

    Sprint 21 Round 3: a line that was NOTHING BUT the marker to begin
    with — LinkedIn's "About the company" widget renders "… more" as
    its own standalone line after each truncated sub-section, not
    glued to visible text the way an earlier, similar case's "...more" was — still
    needs to be dropped here, same as any other all-noise line. This is
    unambiguous to detect: _strip_truncation_markers() only ever
    replaces the matched marker substring and strips whitespace, so its
    result can equal exactly "..." ONLY when the original line had zero
    other content — any real surrounding text would survive into the
    result too. Confirmed via the real anonymized production case
    (tests/fixtures/company_overview_truncation.json) that this is a
    repeated, structural LinkedIn UI pattern (7 standalone "… more"
    lines in that one posting alone), not a one-off. The original glued
    case is untouched: its stripped result is "Designed to scale with
    independent hotels...", never bare "...", so this new check never
    fires for it."""
    cleaned = (_strip_truncation_markers(line) for line in lines)
    return [line for line in cleaned if line and line != "..." and not _UI_CHROME_SENTENCE.search(line)]


def _is_any_section_header(line: str) -> bool:
    if _is_noise_header(line):
        return True
    return any(pattern.match(line) for pattern in _SECTION_HEADERS.values())


def _collect_section_items(
    lines: list[str], start_index: int, *, max_item_length: int = _MAX_SECTION_ITEM_LENGTH,
) -> list[str]:
    """Lines immediately following a recognized section header, up to
    the next recognized header (any of _SECTION_HEADERS or the existing
    _NOISE_HEADERS denylist) or _MAX_SECTION_ITEMS, whichever comes
    first. Accepts both "-"/"•"-prefixed bullets and plain short
    sentences (many postings write requirements as short paragraphs
    rather than a bulleted list) — a generous length cap
    (_MAX_SECTION_ITEM_LENGTH by default) is what keeps a stray long
    paragraph from being swallowed whole as "one requirement".

    Sprint 21 Round 3: `max_item_length` is overridable — every existing
    caller (responsibilities/requirements/preferred/education, via
    _extract_sections()) omits it and keeps the exact original 200-char
    behavior, unchanged. _derive_company_overview() is the one caller
    that passes a larger value: unlike a responsibilities/requirements
    bullet, a company-identity PARAGRAPH is real prose, not a short
    line-item, and a real one (confirmed via the anonymized production case:
    384/434 real characters) can legitimately exceed 200 chars — the
    200-char default was silently dropping the only genuine identity
    candidate before _join_real_sentences()'s content-classification
    filter (Section 18) ever got a chance to prefer it over shorter,
    off-topic content that happened to survive the cap instead."""
    items: list[str] = []
    for line in lines[start_index:]:
        if _is_any_section_header(line) or len(items) >= _MAX_SECTION_ITEMS:
            break
        text = _BULLET_PREFIX.sub("", line).strip()
        if text and len(text) <= max_item_length:
            items.append(text)
    return items


def _extract_sections(lines: list[str]) -> dict[str, list[str]]:
    """Scans for the first occurrence of each known section header and
    collects what follows — see _collect_section_items(). A posting
    missing a section (no "Requirements:" header at all, say) just
    leaves that bucket empty, same "not extracted, never an error"
    contract every other understanding field already has."""
    sections: dict[str, list[str]] = {
        "responsibilities": [], "requirements": [], "preferred": [], "education": [],
    }
    for index, line in enumerate(lines):
        for name, pattern in _SECTION_HEADERS.items():
            if pattern.match(line) and not sections[name]:
                sections[name] = _collect_section_items(lines, index + 1)
                break
    return sections


# Sprint 15.8.11 (Phase 2 Round 2 — Summary semantic redesign): a
# GENERIC, reusable positive signal that a sentence is actually about
# the ROLE — second-person address to the candidate ("you'll drive...",
# "your work will..."), a role/title keyword (_TITLE_KEYWORDS, already
# reviewed and shared with title resolution — not a new list), or the
# resolved title itself appearing in the sentence. Deliberately NOT
# built from any specific reported example's exact wording — "you"/
# "your" is a structural feature of how job postings address candidates
# industry-wide, not a phrase lifted from any one posting.
_ROLE_ADDRESS = re.compile(r"(?i)\byou(?:'ll|'d|'re)?\b|\byour\b")


def _is_role_grounded(sentence: str, title: str) -> bool:
    """Sprint 15.8.11: does `sentence` sound like it's about the role/
    candidate's own work, as opposed to the company's identity or a
    generic hiring pitch? Reused by both _derive_summary()'s candidate
    preference (below) and LLMExtractor's defensive-validation pass
    (llm.py) — one shared signal, not two independently-tuned checks,
    same "shared pattern" convention _UI_CHROME_SENTENCE established
    for noise detection."""
    if _ROLE_ADDRESS.search(sentence):
        return True
    if _TITLE_KEYWORDS.search(sentence):
        return True
    if title and title.strip() and title.strip().lower() in sentence.lower():
        return True
    return False


# Sprint 15.8.11: a SMALL, deliberately generic set of industry-wide
# company-mission/marketing and untargeted-hiring-pitch phrasings —
# "on a mission to", "join our team", "always looking for" are common
# boilerplate used across countless real job postings, not wording
# specific to any one reported example. Used ONLY in combination with
# _is_role_grounded() returning False (see llm.py) as a conservative,
# two-condition trigger — never alone, and never by _derive_summary(),
# which doesn't need a negative list at all (its fallback-to-full-pool
# design already handles "nothing role-grounded found" safely without
# risking a false-positive reject of an otherwise-fine summary).
_GENERIC_COMPANY_PITCH_MARKER = re.compile(
    r"(?i)\b("
    r"on a mission to|our mission is|we believe (?:in|that)|"
    r"global leader in|leading provider of|"
    r"join us\b|join our team\b|"
    r"we(?:'re| are) always looking for"
    r")\b"
)


def _looks_like_generic_company_pitch(text: str) -> bool:
    return bool(_GENERIC_COMPANY_PITCH_MARKER.search(text))


# Sprint 15.8.12 (Phase 2 Round 4 — Company Overview semantic redesign):
# a SEPARATE, SMALL, deliberately generic set of employee-benefits/
# culture/career-program vocabulary — "career growth", "internal
# mobility", "work-life balance", "mentorship" are common HR/benefits
# boilerplate used across countless real job postings' "About Us"/
# "Life at X" page sections, not wording specific to any one reported
# example. Kept as its OWN marker (not folded into
# _GENERIC_COMPANY_PITCH_MARKER above) since it's a genuinely different
# content category — hiring/recruiting pitch ("join our mission") vs.
# benefits/culture programs ("we offer mentorship") — even though both
# ultimately answer "what's it like to work here," not "what is this
# organization and what does it do." _looks_like_non_identity_content()
# below is where the two categories are unified for callers that don't
# need the distinction.
#
# Sprint 21 Round 3: "diversity" widened to "divers(?:e|ity)" — the real
# anonymized production case (tests/fixtures/company_overview_truncation.json)
# has a genuine DEI-section sentence phrased "not just aiming for
# diverse REPRESENTATION" (the adjective, not the noun "diversity"),
# which the noun-only pattern missed entirely. Only reachable once
# Round 3's larger per-item collection cap let this real paragraph
# survive raw collection at all — previously the whole paragraph was
# silently dropped by the 200-char cap before this marker ever got a
# chance to run on it, so this gap existed but was never exercised.
_CULTURE_BENEFITS_MARKER = re.compile(
    r"(?i)\b("
    r"invest in our (?:people|employees|team)|"
    r"career (?:growth|development)|professional development|"
    r"internal mobility|"
    r"work.life balance|"
    r"divers(?:e|ity)(?:,?\s+equity)?(?:,?\s*(?:and|&)\s+inclusion)?|\bDEI\b|"
    r"employee (?:benefits|wellness|resource groups?)|"
    r"mentorship(?:\s+program|\s+opportunit\w+)?|"
    r"paid time off|\bPTO\b|"
    r"health (?:and|&) wellness|"
    r"we (?:offer|provide) (?:a )?(?:comprehensive )?benefits"
    r")\b"
)


def _looks_like_culture_or_benefits_content(text: str) -> bool:
    return bool(_CULTURE_BENEFITS_MARKER.search(text))


def _looks_like_non_identity_content(sentence: str, title: str) -> bool:
    """Sprint 15.8.12: True if `sentence` reads like generic hiring
    pitch, benefits/culture messaging, or role-specific content — none
    of which answers "what is this organization and what does it do,"
    the question Company Overview exists to answer, and none of which
    answers "what will this person work on," the question Summary
    exists to answer either. One shared definition of "off-topic for a
    content-focused understanding field," combining
    _looks_like_generic_company_pitch() (recruiting/mission language),
    _looks_like_culture_or_benefits_content() (HR/benefits/culture
    language — new this sprint), and _is_role_grounded() (role-
    responsibility language — reused here in the OPPOSITE direction
    from its original purpose: a sentence sounding like it's addressing
    the candidate about role duties is exactly what should NOT be in a
    company-identity field, the mirror image of why it's exactly what
    SHOULD be preferred in a role summary). Reused by both
    _join_real_sentences() (company_overview's own candidate
    preference, below) and LLMExtractor's defensive-validation pass
    (llm.py) — one shared place, not three independently-tuned
    checks."""
    return (
        _looks_like_generic_company_pitch(sentence)
        or _looks_like_culture_or_benefits_content(sentence)
        or _is_role_grounded(sentence, title)
    )


def _derive_summary(lines: list[str], title: str) -> Optional[str]:
    """A crude but real summary — the first one or two full sentences
    of real prose, capped to a reasonable length. Deliberately not an
    abstractive summary (that needs an LLM — extraction/llm.py already
    produces one, and is the primary source when that backend is
    enabled); this rule-based version exists as the zero-cost safety-
    layer fallback so the field is never just always-empty. Short,
    heading-shaped fragments (a title/company/location line that
    happens to fall before the first section) are filtered out by a
    minimum word count, not by trying to re-detect which lines are
    headings a second time.

    Sprint 15.7: chrome LINES ("Easy Apply", "245 applicants" — no
    trailing punctuation) are dropped before splitting, and every
    remaining line is sentence-split *individually*, not joined into
    one blob first. A short heading-shaped line (title/company/
    location, also with no trailing punctuation) would otherwise glue
    onto the very next line's real sentence with nothing for
    _SENTENCE_SPLIT to break on between them — one long blob that
    either smuggles the heading text into the "summary" (the min-word-
    count filter can't tell a heading-plus-sentence blob from a real
    long sentence) or, if it happens to contain a chrome phrase
    anywhere, gets the whole blob rejected even though most of it is
    real content. Splitting per line first means a heading fragment
    stays its own short, separately-filterable fragment.

    Sprint 15.8.2: candidate text is no longer always "everything
    before the first header". If that first header is one of
    _INTRO_HEADERS ("About the job" and similar — a header that
    INTRODUCES the posting body, not one that ends an intro like
    "Responsibilities" does), the real prose is what follows it, not
    what precedes it — a posting shaped "chrome, About the job, real
    intro, Responsibilities, ..." previously had nothing before its
    first header worth summarizing, so this always returned None for
    exactly that (extremely common) shape.

    Sprint 15.8.4: that 15.8.2 fix only applied when the intro header
    happened to be the FIRST header found — but a posting can easily
    have "About Us" (or a company card, or other content) BEFORE "About
    the job", and the old logic stopped scanning at whichever header it
    hit first, never reaching the real intro header at all. Now
    actively SEEKS the first _INTRO_HEADERS match anywhere in the text
    (not just checks whether the first header found happens to be one),
    falling back to "everything before the first header of any kind"
    only when no intro header exists anywhere — so "chrome, About Us,
    company mission text, About the job, real intro, Responsibilities"
    now correctly finds and uses the "About the job" content, not the
    company mission text that happened to come first.

    Sprint 15.8.11 (Phase 2 Round 2): finding the right SPAN was never
    the remaining bug — a real reported posting has its company-mission
    paragraph and its role-description paragraph in the exact same
    "About the job" span, one after the other, and pure first-by-
    position selection had no way to prefer one over the other. Once
    the candidate pool of real sentences is built (unchanged from
    above), this now PREFERS sentences that are role-grounded
    (_is_role_grounded()) over the raw first-by-position ones. Falls
    back to the full, unfiltered pool when NONE of it is role-grounded
    — a thin posting with no clearly role-flavored sentence at all
    still gets its previous best-effort behavior, never an empty
    field where one existed before. This is a preference among
    already-found candidates, not a rejection filter: structural
    signals (which span) still come first; semantic filtering only
    re-orders what's already inside that span, exactly the "structural
    signals first, semantic filtering second" split this module's
    other derivations already follow."""
    header_index = len(lines)
    intro_header_index: Optional[int] = None
    for index, line in enumerate(lines):
        if not _is_any_section_header(line):
            continue
        if header_index == len(lines):
            header_index = index
        if intro_header_index is None and line.rstrip(":?!").strip().lower() in _INTRO_HEADERS:
            intro_header_index = index

    if intro_header_index is not None:
        content_end = len(lines)
        for index in range(intro_header_index + 1, len(lines)):
            if _is_any_section_header(lines[index]):
                content_end = index
                break
        candidate_lines = lines[intro_header_index + 1:content_end]
    else:
        candidate_lines = lines[:header_index]

    # A bare heading-shaped line (title/company/location/department —
    # no terminal punctuation, e.g. "Senior Machine Learning Engineer -
    # LLMs") can coincidentally cross _MIN_SUMMARY_SENTENCE_WORDS once
    # split on its own (a dash counts as a "word") — reuse
    # _looks_like_heading() (already used elsewhere in this module for
    # exactly "is this a heading, not prose" checks) to exclude it
    # rather than let word count alone decide. A genuine prose sentence
    # almost always ends in terminal punctuation, which
    # _looks_like_heading() already treats as disqualifying.
    intro_lines = [
        line for line in _strip_ui_noise_lines(candidate_lines)
        if not _looks_like_heading(line)
    ]
    sentences: list[str] = []
    for line in intro_lines:
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip())
    real_sentences = [
        s for s in sentences
        if len(s.split()) >= _MIN_SUMMARY_SENTENCE_WORDS and not _UI_CHROME_SENTENCE.search(s)
    ]
    if not real_sentences:
        return None
    role_grounded = [s for s in real_sentences if _is_role_grounded(s, title)]
    selected = role_grounded if role_grounded else real_sentences
    summary = " ".join(selected[:2]).strip()
    return summary[:500] or None


def _join_real_sentences(items: list[str], title: str) -> Optional[str]:
    """Sprint 15.8.4: factored out of _derive_company_overview() so
    _guess_company_card_tagline() (below) can reuse the exact same
    logic rather than a second copy of it — split each item into
    sentences *individually* (never glue two lines into one blob before
    splitting; a mid-sentence line break must not be lost or truncated,
    same reasoning as _derive_summary()'s per-line splitting), drop
    anything that still trips the shared noise filter, join everything
    that's left. Unlike _derive_summary() (which deliberately caps at
    the first 1-2 sentences, since that pool can contain unrelated
    heading fragments), this joins every remaining sentence — the
    callers here already scope the candidate pool tightly (an explicit
    section, or a small bounded window after a company card), so there
    is no similar risk of accidentally running on.

    Sprint 15.8.8: strips a "…more"/"...more" truncation-link label
    from each sentence — see _strip_truncation_markers()'s docstring.
    Applied here (not just in _strip_ui_noise_lines(), which some but
    not all callers of this function run their input through first)
    so both consumers get it regardless of what preprocessing already
    happened to `items`.

    Sprint 15.8.12 (Phase 2 Round 4): also PREFERS sentences that don't
    read as generic hiring pitch, benefits/culture content, or role-
    responsibility content (_looks_like_non_identity_content()) over
    ones that do — a real reported "About Us"-adjacent section mixed
    genuine company-identity sentences ("founded in 2012...") with
    benefits/culture sentences ("we offer mentorship...") under the
    same header, and nothing previously distinguished them. Same
    "prefer, with graceful fallback" shape _derive_summary()'s own
    candidate preference already established (Sprint 15.8.11): falls
    back to the FULL sentence pool when every sentence is flagged,
    rather than returning an empty/near-empty field where one existed
    before — a thin "About Us" section that's entirely benefits-
    flavored still gets its previous best-effort content, never
    nothing. Applied HERE (this function), not inside
    _derive_company_overview() directly, so both of this function's
    callers — the explicit-header path and the company-card-tagline
    fallback — share one filtering pass, not two."""
    sentences: list[str] = []
    for item in items:
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(item) if s.strip())
    cleaned_sentences = [_strip_truncation_markers(s) for s in sentences]
    real_sentences = [s for s in cleaned_sentences if s and not _UI_CHROME_SENTENCE.search(s)]
    if not real_sentences:
        return None
    on_topic = [s for s in real_sentences if not _looks_like_non_identity_content(s, title)]
    selected = on_topic if on_topic else real_sentences
    return " ".join(selected)[:500] or None


_COMPANY_CARD_TAGLINE_WINDOW = 5  # how many lines past the card block to scan for a description


def _guess_company_card_tagline(lines: list[str], title: str) -> Optional[str]:
    """Sprint 15.8.4: a company card can bundle a short description
    directly below its stats (name, followers, industry, employee
    count) with no separate "About Us"-style header at all — a
    real, reported LinkedIn shape, and a generic one (many company-
    profile-card widgets pair a one-line tagline with their stats),
    not a per-site assumption. Only consulted as a FALLBACK by
    _derive_company_overview() below, when no explicit header exists —
    an explicit "About Us"/etc. section always wins when present.

    Sprint 15.8.5: the END of the card block (where a tagline would
    begin) now comes from the SAME _find_company_card_block() both
    this function and _guess_company_from_card() use — one shared
    understanding of the card's extent, not two independently-scanning
    implementations that could disagree about it (this sprint's
    explicit "company resolution and company overview must never
    disagree" requirement). Whatever real, sentence-like prose follows
    within a small bounded window (_COMPANY_CARD_TAGLINE_WINDOW lines)
    becomes the candidate — stopping early at any noise line (a
    recruiter/follow prompt), a location line, a workplace-type line,
    or the job title itself, so none of those can ever be mistaken for
    a company description (this sprint's explicit validation
    requirement).

    `title` (Sprint 15.8.5) is the caller's already-resolved job
    title, not a re-guess — same reasoning as
    _guess_company_from_card(). Reproduced directly: when the card
    sits directly above the title with no header between them (the
    exact "company card before the job title" layout this sprint's
    report describes), the tagline window's forward scan had nothing
    stopping it at the title line — a heading-shaped line that's
    neither a location nor a workplace-type value, so the existing
    guard didn't catch it — and the title text got glued onto the end
    of the company overview."""
    if not lines:
        return None
    first_anchor_index: Optional[int] = None
    for anchor_index, line in enumerate(lines[:_COMPANY_CARD_SCAN_WINDOW]):
        if _is_company_card_anchor(line):
            first_anchor_index = anchor_index
            break
    if first_anchor_index is None:
        return None

    index = _find_company_card_block(lines, first_anchor_index, title).end_index

    candidate_lines: list[str] = []
    for line in lines[index:index + _COMPANY_CARD_TAGLINE_WINDOW]:
        if _is_any_section_header(line):
            break
        if _line_matches_title(line, title):
            break  # the job title itself — never part of a company description
        # Validation (Sprint 15.8.4): a tagline candidate must never be
        # a noise/recruiter-prompt line, a location line, or a bare
        # workplace-type label — reuses the exact same checks
        # _plausible_company_name() already applies to a company-
        # NAME candidate, applied here to a company-DESCRIPTION
        # candidate instead. The location/workplace check is only
        # applied to a HEADING-shaped line (short, no terminal
        # punctuation) — a real location/workplace value always looks
        # like that; a real description sentence never does, and
        # _CITY_COMMA's pattern (city+comma+more-text) can otherwise
        # false-positive on an ordinary sentence that happens to
        # contain exactly one comma (e.g. "We are Acme Technologies,
        # simply known as Acme." — no city involved at all).
        if _UI_CHROME_SENTENCE.search(line):
            break
        if _looks_like_heading(line) and (_looks_like_location_line(line) or _WORKPLACE_TYPE_LINE.match(line)):
            break
        candidate_lines.append(line)

    if not candidate_lines:
        return None
    return _join_real_sentences(candidate_lines, title)


def _derive_company_overview(lines: list[str], title: str) -> Optional[str]:
    """Sprint 15.7: a company_overview populated ONLY from the posting's
    own "About Us"/"About the Company"/"Company Overview"/"Who We Are"
    text, when present — reusing _COMPANY_OVERVIEW_HEADERS (the
    company-related subset of the existing _NOISE_HEADERS denylist) to
    find where that section starts, then collecting what follows the
    same way _collect_section_items() does for responsibilities/
    requirements, joined as prose (this is a paragraph, not a bullet
    list) and capped like _derive_summary(). Deliberately never reaches
    out to any external source — a prior sprint researched and declined
    to build a "look this company up on the internet" feature (no
    infrastructure for it exists, and none is added here); this only
    reads what the employer already wrote about themselves in the
    posting.

    Sprint 15.8.4: falls back to _guess_company_card_tagline() when no
    explicit header exists at all — a real reported shape where the
    company's own description sits directly below its card, with
    nothing labeling it. The explicit-header path always wins when a
    genuine section is present; the card tagline only fires when
    there's truly nothing else to go on. None when neither source
    finds anything — never invented.

    Sprint 15.8.8: also drops any heading-shaped item, the same way
    _derive_summary() already filters its own candidate pool — a real
    reported example (a company's "About Us"-equivalent page section
    spanning multiple mini-sections: an overview blurb, then separate
    "Commitments"/"Career growth and learning"-style sub-headers each
    with their own blurb) showed _collect_section_items() has no way
    to stop at a sub-header it doesn't recognize (_NOISE_HEADERS/
    _SECTION_HEADERS is a fixed, reviewed list — "Commitments" was
    never going to be in it, and adding every possible company-page
    sub-header name would be exactly the open-ended heuristic list
    this codebase avoids). _collect_section_items() itself is shared
    with the bulleted responsibilities/requirements sections, where a
    short heading-shaped ITEM is often the real content (e.g. "5+
    years of experience") — so the filter can't live there without
    breaking that. It's safe here specifically because this call site
    collects PROSE, not bullets: a genuine "About Us" paragraph is
    written in full sentences (terminal punctuation), so a heading-
    shaped fragment mixed into that pool is far more likely a sub-
    header than real content — the exact same reasoning
    _derive_summary() already relies on for its own prose collection."""
    header_index: Optional[int] = None
    for index, line in enumerate(lines):
        if line.rstrip(":?!").strip().lower() in _COMPANY_OVERVIEW_HEADERS:
            header_index = index
            break
    if header_index is None:
        return _guess_company_card_tagline(lines, title)

    # Sprint 21 Round 3: a larger per-item cap than the 200-char default
    # every other _collect_section_items() caller keeps — see that
    # function's own docstring. 500, not an arbitrary new number: the
    # same final-output cap _join_real_sentences() below already applies
    # to this field, so a single real identity paragraph is never
    # allowed to survive raw collection only to be truncated later
    # anyway; it also matches _derive_summary()'s own 500-char cap
    # (line ~1823), keeping every prose field's size budget consistent.
    raw_items = _collect_section_items(lines, header_index + 1, max_item_length=500)
    items = [line for line in _strip_ui_noise_lines(raw_items) if not _looks_like_heading(line)]
    if not items:
        return None
    return _join_real_sentences(items, title)


# Reuses the existing skill catalog (catalog.py) and matcher
# (_skills.py's build_skill_engine()) rather than a second,
# extraction-specific skill list. catalog.py's own docstring already
# documents it as
# user-editable ("Replace with whatever skills are relevant to the
# roles you personally track"); this only wires that existing,
# intentionally-small catalog into extraction too, it doesn't grow it.
_SKILL_PATTERN, _SKILL_CANONICAL = build_skill_engine(TRACKED_SKILLS)


def _find_catalog_skills(text: str) -> list[str]:
    seen: set[str] = set()
    found: list[str] = []
    for match in _SKILL_PATTERN.finditer(text):
        canonical = _SKILL_CANONICAL.get(match.group(1).lower(), match.group(1))
        if canonical not in seen:
            seen.add(canonical)
            found.append(canonical)
    return found


class _Understanding(NamedTuple):
    summary: Optional[str]
    responsibilities: list[str]
    requirements: list[str]
    required_skills: list[str]
    preferred_skills: list[str]
    education_requirements: list[str]
    company_overview: Optional[str]


def _extract_understanding(description: str, title: str) -> _Understanding:
    """The one place extract_from_text()/extract_from_url() both call
    for summary/responsibilities/requirements/required_skills/
    preferred_skills/education_requirements/company_overview — see this
    module's own docstring for why neither duplicates this logic
    separately. "Preferred qualifications" (one of this project's
    understanding fields) has no dedicated bucket on ExtractionResult/
    a downstream persistence layer (see base.py) — its text joins `requirements`
    (a preferred qualification is still a qualification the posting
    asks for, just an optional one), while any catalog skill mentioned
    specifically within that section is what populates
    `preferred_skills` — reusing the existing field rather than
    inventing a new one, per this sprint's "don't change backend models
    unnecessarily" guidance.

    Sprint 15.7 adds education_requirements (a distinct "Education:"
    section, kept ALSO in `requirements` — deliberately additive/
    duplicated rather than moved out, so nothing that already reads
    `requirements` for match/advise scoring changes behavior) and
    company_overview (see _derive_company_overview()'s own docstring
    for why this is never an external lookup)."""
    lines = _clean_lines(description)
    sections = _extract_sections(lines)
    summary = _derive_summary(lines, title)
    company_overview = _derive_company_overview(lines, title)

    preferred_text = "\n".join(sections["preferred"])
    preferred_skills = _find_catalog_skills(preferred_text) if preferred_text else []
    required_skills = [
        skill for skill in _find_catalog_skills(description) if skill not in preferred_skills
    ]

    return _Understanding(
        summary=summary,
        responsibilities=sections["responsibilities"],
        requirements=sections["requirements"] + sections["preferred"] + sections["education"],
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        education_requirements=sections["education"],
        company_overview=company_overview,
    )


def _source_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    for known in ("linkedin.com", "greenhouse.io", "lever.co", "indeed.com"):
        if host.endswith(known):
            return known.split(".")[0]
    return host or "web"


def looks_like_url(text: str) -> bool:
    """True if `text` parses as an http(s) URL — used by the CLI to tell a
    pasted URL apart from a pasted job description."""
    parsed = urlparse(text)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


_SOURCE_LABELS = {
    "linkedin": "LinkedIn URL",
    "greenhouse": "Greenhouse URL",
    "lever": "Lever URL",
    "indeed": "Indeed URL",
}


def describe_source(url: str) -> str:
    """Human-friendly label for a job URL's site, for CLI prompts (e.g.
    "Detected: LinkedIn URL")."""
    return _SOURCE_LABELS.get(_source_from_url(url), "Job URL")


class RuleBasedExtractor(JobExtractor):
    """
    Deterministic JobExtractor: line-based pattern matching over pasted
    text (Title:/Company:/Location: fields, "As a ROLE" sentences, a
    LinkedIn-style structured header) or, for a URL, the page's schema.org
    JobPosting JSON-LD / Open Graph metadata. No LLM, no dependency beyond
    the standard library.
    """

    def extract_from_text(
        self, text: str, source_url: Optional[str] = None, *, structured_hints: Optional[dict] = None,
    ) -> ExtractionResult:
        """
        Turn pasted/file job-description text into an ExtractionResult.

        Extraction priority:
          0. Structured hints (Sprint 15.7) — a schema.org JobPosting
             JSON-LD node forwarded by a browser client (see
             _parse_structured_hints()) — HIGH confidence, same as
             extract_from_url()'s own JSON-LD priority. `og:site_name`
             (Open Graph) is used as a company-only fallback when the
             JSON-LD is absent or has no company.
          1. Structured header ("Staff ML Engineer - Advertising" followed
             by "Amsterdam, North Holland, Netherlands") — the site's own
             rendered header, when the paste starts with one.
          2. Explicit patterns in the description ("Title:"/"Company:"
             fields, "As a ROLE ..." / "We are looking for a ROLE..."
             sentences, department clauses like "within our X team").
          3. Fallback guessing (a heading-shaped line with no other
             signal).

        Anything not confidently identified falls back to "Unknown"/LOW.
        The full input text is kept as `description` unmodified so skill
        matching always sees the complete original text, regardless of
        title confidence — but summary/responsibilities/requirements/
        required_skills/preferred_skills (Sprint 15.7) are derived from
        whichever text is richest: the JSON-LD `description` when one
        was forwarded (typically the full, clean posting — no page
        chrome noise), otherwise this same input text.
        """
        stripped = text.strip()
        if not stripped:
            raise ValueError("raw_text is empty — nothing to extract")

        lines = _clean_lines(stripped)
        posting, meta = _parse_structured_hints(structured_hints)
        header = _guess_structured_header(lines)

        if posting and posting.get("title"):
            title = str(posting["title"]).strip()
            confidence = TitleConfidence.HIGH
        elif header:
            title = header[0]
            confidence = TitleConfidence.HIGH
        else:
            title, confidence = _guess_title_with_confidence(lines)

        # Sprint 15.8.3: the shared JSON-LD > og:site_name > company-card
        # > text-heuristics cascade — see _resolve_company()'s own
        # docstring for the full priority order.
        company = _resolve_company(lines, posting, meta, title) or "Unknown"

        location = (posting and _location_from_jsonld(posting)) or None
        if not location or location == "Unknown":
            location = header[2] if header else _guess_location(lines, title)

        department = (
            (posting and _department_from_jsonld(posting))
            or (header[1] if header else None)
            or _guess_department(lines)
        )

        # Sprint 15.8.1: JSON-LD only (Priority 1), no text-fallback for
        # employment_type/posted_at/company_url — see this file's own
        # section docstring just above _employment_type_from_jsonld()
        # for why.
        employment_type = (posting and _employment_type_from_jsonld(posting)) or None
        posted_at = (posting and _posted_at_from_jsonld(posting)) or None
        company_url = (posting and _company_url_from_jsonld(posting)) or None
        # Sprint 15.8.4: workplace_type gains a text-heuristic fallback
        # (a standalone "Remote"/"Hybrid"/"On-site" line) — JSON-LD
        # stays Priority 1 and is NEVER overridden by it; the text
        # heuristic only fills in a value JSON-LD didn't provide.
        workplace_type = (posting and _workplace_type_from_jsonld(posting)) or _guess_workplace_type_from_text(lines, title)

        # Prefer the JSON-LD description when present and substantial —
        # it's typically the full, clean posting text (no nav/chrome
        # noise an innerText-based browser extraction can pick up),
        # which makes the section/summary/skill extraction below
        # meaningfully more accurate. A short/near-empty JSON-LD
        # description (some employers under-populate it) falls back to
        # the input text rather than risking an even-thinner result.
        understanding_source = stripped
        if posting and posting.get("description"):
            jsonld_description = _strip_html(str(posting["description"]))
            if len(jsonld_description) >= 100:
                understanding_source = jsonld_description
        understanding = _extract_understanding(understanding_source, title)

        # A URL is still the strongest identity key when one is known;
        # text identity is only the fallback for file/pasted input.
        job_id = _job_id_for(source_url) if source_url else _job_id_for(
            _identity_signature(title, company, stripped)
        )

        job = Job(
            job_id=job_id,
            title=title,
            company=company,
            location=location,
            url=source_url or f"local://pasted/{job_id}",
            source="linkedin" if source_url else "pasted",
            description=stripped,
            posted_at=posted_at,
            scraped_at=datetime.now(timezone.utc),
        )
        return ExtractionResult(
            job=job, title_confidence=confidence, department=department,
            summary=understanding.summary,
            employment_type=employment_type,
            responsibilities=understanding.responsibilities,
            requirements=understanding.requirements,
            required_skills=understanding.required_skills,
            preferred_skills=understanding.preferred_skills,
            education_requirements=understanding.education_requirements,
            company_overview=understanding.company_overview,
            workplace_type=workplace_type,
            company_url=company_url,
        )

    def extract_from_url(self, url: str) -> ExtractionResult:
        """
        Fetch a job posting URL and turn it into an ExtractionResult.

        Tries, in order: the page's schema.org JobPosting JSON-LD block
        (most reliable — title/company/location/description all come from
        structured data, so its title is HIGH confidence — this is
        Priority 2, the URL path's structured source), then Open Graph
        meta tags or the <title> tag (page-authored but unstructured —
        MEDIUM, Priority 3), then "Untitled" if nothing usable was found
        at all (LOW, Priority 4). Department comes from JSON-LD when an
        employer includes one, else falls back to the same "within our X
        team" pattern used for pasted text (Priority 3), scanned over the
        page's visible text.
        """
        html = fetch_job_page(url)
        parser = _PageParser()
        parser.feed(html)

        posting: Optional[dict] = None
        for block in parser.jsonld_blocks:
            block = block.strip()
            if not block:
                continue
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                continue
            posting = _find_jobposting(data)
            if posting:
                break

        # Sprint 15.8.3: the same shared company-resolution cascade
        # extract_from_text() uses — see _resolve_company()'s docstring.
        # Scanned against the page's own raw visible text (parser.text),
        # since a company card is page content, not something that
        # would ever appear inside a JSON-LD description field. This
        # also closes a pre-existing gap: extract_from_url() previously
        # had no text-heuristic company fallback at all (only JSON-LD),
        # unlike extract_from_text() — both entry points now agree.
        page_lines = _clean_lines(parser.text)

        if posting and posting.get("title"):
            title = str(posting["title"])
            confidence = TitleConfidence.HIGH
            company = _resolve_company(page_lines, posting, parser.meta, title) or "Unknown"
            location = _location_from_jsonld(posting)
            description = _strip_html(str(posting.get("description", ""))) or parser.text
        else:
            title = parser.meta.get("og:title") or parser.title or ""
            confidence = TitleConfidence.MEDIUM if title.strip() else TitleConfidence.LOW
            title = title or "Untitled"
            company = _resolve_company(page_lines, posting, parser.meta, title) or "Unknown"
            location = _location_from_jsonld(posting) if posting else "Unknown"
            description = (
                (_strip_html(str(posting.get("description", ""))) if posting else "")
                or parser.meta.get("og:description")
                or parser.text
            )

        if not description.strip():
            raise JobFetchError(url, "unavailable page", "no usable job content found")

        department = _department_from_jsonld(posting) if posting else None
        if department is None:
            department = _guess_department(_clean_lines(description))

        # Sprint 15.8.1: same JSON-LD-only, no-text-fallback contract as
        # extract_from_text() for employment_type/posted_at/company_url
        # — see this file's section docstring above
        # _employment_type_from_jsonld().
        employment_type = _employment_type_from_jsonld(posting) if posting else None
        posted_at = _posted_at_from_jsonld(posting) if posting else None
        company_url = _company_url_from_jsonld(posting) if posting else None
        # Sprint 15.8.4: same JSON-LD-priority-preserved text fallback
        # as extract_from_text() — see that method's own comment.
        workplace_type = (
            (_workplace_type_from_jsonld(posting) if posting else None)
            or _guess_workplace_type_from_text(page_lines, title)
        )

        # Sprint 15.7: same understanding extraction extract_from_text()
        # uses — see _extract_understanding()'s own docstring for why
        # this isn't a second implementation.
        understanding = _extract_understanding(description, title)

        job = Job(
            job_id=_job_id_for(url),
            title=title.strip(),
            company=company.strip(),
            location=location.strip(),
            url=url,
            source=_source_from_url(url),
            description=description.strip(),
            posted_at=posted_at,
            scraped_at=datetime.now(timezone.utc),
        )
        return ExtractionResult(
            job=job, title_confidence=confidence, department=department,
            summary=understanding.summary,
            employment_type=employment_type,
            responsibilities=understanding.responsibilities,
            requirements=understanding.requirements,
            required_skills=understanding.required_skills,
            preferred_skills=understanding.preferred_skills,
            education_requirements=understanding.education_requirements,
            company_overview=understanding.company_overview,
            workplace_type=workplace_type,
            company_url=company_url,
        )
