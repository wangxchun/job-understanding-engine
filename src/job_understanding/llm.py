"""
LLM-based JobExtractor backend using the Anthropic API.

Same JobExtractor interface and ExtractionResult contract as
RuleBasedExtractor (extraction/rule_based.py) — this is an alternative
implementation behind the same abstraction, not a replacement:

    JobExtractor
        |
   ----------------
   |              |
RuleBased       LLMExtractor  (this file)

Requires the optional `llm` extra (`pip install -e ".[llm]"`) and an
ANTHROPIC_API_KEY in the environment; the `anthropic` package is imported
lazily (inside _build_client(), never at module import time) so it is
never a mandatory dependency for job_tracker — importing this module, or
even the whole extraction/ package, works fine without it installed.

URL extraction reuses rule_based.fetch_job_page() and rule_based._PageParser
for the actual HTTP fetch + HTML parsing — this module only adds an LLM
extraction step on top of already-fetched page text; it never
re-implements scraping.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

from .schema import Job

from ._config import get_settings
from .base import (
    ExtractionResult, JobExtractor, JobFetchError, TitleConfidence,
)
# Imported as a module (not `from ... import fetch_job_page`) so calls go
# through rule_based.fetch_job_page at call time — the same attribute
# tests monkeypatch for RuleBasedExtractor's URL-fetch tests transparently
# intercepts LLMExtractor's URL fetch too, with one patch point for both.
from . import rule_based

__all__ = ["LLMExtractor", "LLMExtractionError"]

# "summary" is intentionally asked for as a short summary, not a verbatim
# copy: the caller already has the full original text (the input to
# extract_from_text, or the fetched page text for a URL) and uses that
# directly for Job.description — see _extract() below. Round-tripping a
# long posting through the model as JSON would cost extra output tokens
# and risks truncation/mangling for no benefit.
#
# Sprint 9.1 ("Job Understanding Layer"): the schema grew from a plain
# parser's fields (title/company/department/location) into structured job
# intelligence — seniority, required vs. preferred skills, and a
# tech_stack distinct from the broader skills list. required_skills is
# what the posting states as necessary ("must have", "required",
# unqualified bullet requirements); preferred_skills is explicitly
# optional/"nice to have" language. tech_stack is the narrower set of
# concrete technology names (languages, frameworks, tools, platforms)
# mentioned anywhere in the posting, which typically overlaps with but
# is not identical to required/preferred_skills (a "skill" can be
# non-technical, e.g. "stakeholder communication").
#
# Sprint 9.3 ("Structured Job Understanding") adds responsibilities/
# requirements as their own list-of-strings fields, distinct from
# required_skills: responsibilities are what the role *does*
# day-to-day (duties/activities), requirements are the qualifications/
# experience/credentials asked of the candidate (which required_skills
# is a technical-skill-flavored subset of) — kept as separate bullet
# lists rather than folded into "summary" so future consumers can use
# them as structured data, not prose to re-parse.
_SYSTEM_PROMPT = """You extract structured job posting data from raw text.

The input is often extracted directly from a live job board page (e.g.
via a browser extension), not a clean pasted job description. It may
contain, mixed in with the real posting, any of:
- navigation and page-chrome text (menus, buttons, "Show more"/"Show less")
- application/activity metadata (apply-button labels, applicant or
  click counts like "94 applicants" or "Over 100 people clicked apply",
  "Easy Apply", "Promoted by hirer", "Reposted N ago")
- recommendation/comparison widgets ("People also viewed", "Similar
  jobs", "Set alert for similar jobs", "See how you compare to other
  applicants", "Your profile is missing some qualifications", "Based on
  LinkedIn data")
- a company-profile card (a company name followed by its own follower
  count, "Follow" button, industry, and employee count — e.g. "77,037
  followers", "1001-5000 employees", "1,044 on LinkedIn") and any
  recruiter/follow prompt near it ("Interested in working with us in
  the future?", "Members who share that they're interested get
  notified about new jobs.")
Treat all of the above as noise, not job content — never repeat a
follower count, employee count, "Follow" label, or recruiter prompt in
any field's value, even inside "company_overview".

Distinguish four separate kinds of text in the input: JOB content
(responsibilities, requirements, role purpose — goes in
summary/responsibilities/requirements/required_skills/etc.), COMPANY
IDENTITY content (what the company makes/does, its industry, its
scale or history — e.g. under "About Us" or alongside a company card
— goes only in company_overview), CULTURE/BENEFITS content (employee
benefits, DEI/culture statements, career-growth or mentorship
programs, work-life balance — often sits in the same "About Us"-style
section as company identity content, but goes in NEITHER summary nor
company_overview; it describes what it's like to work there, not what
the role does or what the organization is), and PAGE METADATA
(everything in the noise categories above — goes in no field at all).
A sentence describing what the company does belongs in
company_overview, never in summary; a sentence describing what the
role does belongs in summary, never in company_overview; a sentence
describing employee benefits or culture belongs in neither.

Return ONLY a single JSON object — no markdown fences, no commentary — matching exactly this shape:

{
  "title": string,
  "company": string,
  "department": string or null,
  "location": string,
  "employment_type": string or null,
  "seniority": string or null,
  "summary": string,
  "responsibilities": array of strings,
  "requirements": array of strings,
  "required_skills": array of strings,
  "preferred_skills": array of strings,
  "tech_stack": array of strings,
  "education_requirements": array of strings,
  "company_overview": string or null
}

Rules:
- "title" is the job title alone. Never fold in a department/team qualifier — e.g. for "Machine Learning Engineer within our Advertising team", title is "Machine Learning Engineer" and department is "Advertising", not the whole phrase.
- "title" and "company" must never be the same string, and "company" must never itself look like a job title — never return the role name as the company. If the company genuinely cannot be determined from the text, use "Unknown" rather than guessing from the title or inventing one.
- Use "Unknown" for company/location if genuinely not stated in the text; use null for department/employment_type/seniority if not stated.
- "employment_type" is e.g. "Full-time", "Part-time", "Contract", "Internship" — only if stated.
- "location" is a geographic place (city/region/country, or "Remote" only if that's genuinely the entire location with no city given) — never include the work arrangement (Remote/Hybrid/On-site) as part of it when a real geographic location is also stated; e.g. for "Amsterdam Area" and, separately, "Hybrid", location is "Amsterdam Area", not "Hybrid" and not "Amsterdam Area, Hybrid".
- "seniority" is the role's level if stated or clearly implied by the title (e.g. "Junior", "Mid", "Senior", "Staff", "Principal", "Lead", "Intern") — null if genuinely ambiguous, never guessed from years-of-experience arithmetic.
- "summary" is a concise 1-3 sentence summary of what the role does and why it exists (its technical/business purpose), prioritizing an explicit "About the job"/"About the role"/"Role overview"/"Position overview" section if the posting has one. Genuinely summarize and rewrite in your own words, never copy a raw sentence or paragraph verbatim from the input. Never summarize the company's mission, an "About Us" section, company culture, or benefits — those describe the company, not the role, and belong in company_overview (if anywhere), never in summary. Also exclude generic hiring/recruiting pitch that isn't tied to this role's specific duties (e.g. "we're always looking for talented, like-minded people", "join our mission", "help us shape the future") — that's an invitation to apply, not a description of the work. If a posting's intro section opens with company mission or generic recruiting language before getting to what the role actually involves, look past that opening for the role-specific content (e.g. a "responsibilities"/"what you'll do" section, or a later paragraph that names concrete duties) rather than summarizing the opening. When an intro section contains BOTH company/mission content and role-specific content, summarize only the role-specific part. Never include any of the noise categories described above.
- "responsibilities" lists what the role actually does day-to-day (e.g. "Build ranking models", "Own the deployment pipeline") — short, individual bullet-style strings, not one long paragraph.
- "requirements" lists qualifications/experience/credentials the posting asks for as plain statements (e.g. "5+ years of software engineering experience", "BS in Computer Science or related field") — distinct from required_skills below, which is specifically the skill/technology names, not the full requirement sentence. Education/degree requirements belong here too, in addition to education_requirements below.
- "required_skills" lists skills/qualifications the posting states as necessary (e.g. "must have", "required", plain requirement bullets) — can include non-technical skills (e.g. "stakeholder communication"), not only technologies.
- "preferred_skills" lists skills the posting explicitly marks as optional/nice-to-have/a plus — never put a required skill here.
- "tech_stack" lists concrete technology names mentioned anywhere in the posting (languages, frameworks, libraries, tools, cloud platforms — e.g. "Python", "AWS", "Kubernetes"), regardless of whether they're required or preferred.
- "education_requirements" lists degree/certification/academic requirements stated in the posting (e.g. "BS in Computer Science or related field") — a structured, separately-displayed subset of what's also included in requirements above, not a replacement for it.
- "company_overview" is a 1-3 sentence description of what the company does/its industry, drawn ONLY from the posting's own explicit "About Us"/"About the Company"/"Who We Are"-style section if the posting includes one — null if it doesn't. Focus only on company identity: what the company makes or does, its industry/domain, and factual scale or history (founding year, size, geographic reach). Exclude employee benefits, culture/DEI statements, career-growth or mentorship programs, and work-life-balance language, even when they appear within the same "About Us"-style section — those describe why someone should want to work there, not what the organization is or does, and belong in neither this field nor summary. If an "About Us"-style section mixes company-identity sentences with benefit/culture sentences, include only the identity sentences. Never draw this from a recommendation/comparison/analytics widget even if it happens to mention the company's name; never invent, infer, or supply company information from your own outside knowledge — this field reflects only what the posting itself explicitly states about the company, nothing else.
- If the input is not a job posting at all, still return the JSON shape with best-effort/"Unknown"/empty-array/null values — never refuse, never add prose outside the JSON object.
"""

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "company": {"type": "string"},
        "department": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "location": {"type": "string"},
        "employment_type": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "seniority": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "summary": {"type": "string"},
        "responsibilities": {"type": "array", "items": {"type": "string"}},
        "requirements": {"type": "array", "items": {"type": "string"}},
        "required_skills": {"type": "array", "items": {"type": "string"}},
        "preferred_skills": {"type": "array", "items": {"type": "string"}},
        "tech_stack": {"type": "array", "items": {"type": "string"}},
        "education_requirements": {"type": "array", "items": {"type": "string"}},
        "company_overview": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": [
        "title", "company", "department", "location", "employment_type",
        "seniority", "summary", "responsibilities", "requirements",
        "required_skills", "preferred_skills", "tech_stack",
        "education_requirements", "company_overview",
    ],
    "additionalProperties": False,
}

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class LLMExtractionError(RuntimeError):
    """
    Raised when the LLM backend can't be used at all — the `anthropic`
    package isn't installed, no API credentials are configured, or the API
    request itself fails (network, auth, rate limit, ...).

    Malformed/incomplete *model output* is deliberately NOT raised as an
    error — see _call_llm(), which degrades to TitleConfidence.LOW instead
    so the normal human-in-the-loop confirmation flow (main.py) handles it,
    the same as a low-confidence RuleBasedExtractor guess.
    """


def _require_anthropic():
    try:
        import anthropic
    except ImportError as exc:
        raise LLMExtractionError(
            "The anthropic package is required for LLM-based extraction. "
            "Install with: pip install -e '.[llm]'"
        ) from exc
    return anthropic


def _build_client():
    """
    Build a real Anthropic client. If Settings resolved an
    ANTHROPIC_API_KEY (env var or .env, see _config.get_settings()),
    pass it explicitly; otherwise fall back to bare
    `Anthropic()`, which resolves credentials from the environment on
    its own (ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN).

    If neither is configured anywhere the SDK would look,
    this raises LLMExtractionError with a clear, actionable message
    *before* attempting a network call, instead of letting the SDK's own
    request-time AuthenticationError ("Could not resolve authentication
    method...") be the first and only signal — that message never says
    which environment variable to set. Never includes the key's value in
    either case — there is nothing to leak when it's absent, and the
    present case never surfaces it.
    """
    anthropic = _require_anthropic()
    api_key = get_settings().anthropic_api_key
    if not api_key and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        raise LLMExtractionError(
            "Anthropic API key is not configured. Please set ANTHROPIC_API_KEY "
            "in your environment."
        )
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    return anthropic.Anthropic()


def _extract_json_object(raw_text: str) -> Any:
    """
    Parse the model's JSON response, tolerating stray markdown fences or
    leading/trailing prose a model might add despite instructions not to.
    Raises json.JSONDecodeError on genuinely malformed input — callers
    catch that and degrade to LOW confidence rather than crashing.
    """
    stripped = raw_text.strip()
    match = _JSON_OBJECT.search(stripped)
    candidate = match.group(0) if match else stripped
    return json.loads(candidate)


def _clean_str(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _clean_optional_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _clean_str_list(value: Any) -> list[str]:
    """Coerce a model-returned array field to a list of non-empty
    strings — tolerates a missing/null/wrong-typed field (older cached
    responses, a model slipping outside the schema) the same way the
    scalar _clean_* helpers do, degrading to [] rather than raising."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _with_structured_hint_context(text: str, structured_hints: Optional[dict]) -> str:
    """Sprint 15.7: prepends a short, human-readable summary of any
    JSON-LD/meta hints a browser client forwarded (see
    rule_based._parse_structured_hints(), reused here rather than a
    second JSON-LD parser) to the text sent to the model — extra
    context only, never a replacement for the description text itself.
    A capable model already infers title/company well from raw text
    most of the time; this mainly helps when a page's own visible text
    is noisy (LinkedIn's chrome-heavy innerText, e.g.) but its
    structured metadata is clean. Returns `text` unchanged when there's
    nothing usable to add — the common case for a plain CLI paste,
    which never has structured_hints to begin with."""
    if not structured_hints:
        return text

    posting, meta = rule_based._parse_structured_hints(structured_hints)
    facts: list[str] = []
    if posting:
        title = posting.get("title")
        if isinstance(title, str) and title.strip():
            facts.append(f"Title (from page metadata): {title.strip()}")
        company = rule_based._company_from_jsonld(posting)
        if company != "Unknown":
            facts.append(f"Company (from page metadata): {company}")
        location = rule_based._location_from_jsonld(posting)
        if location != "Unknown":
            facts.append(f"Location (from page metadata): {location}")
    elif meta.get("og:site_name"):
        facts.append(f"Company (from page metadata): {meta['og:site_name']}")

    if not facts:
        return text

    hint_block = (
        "Known facts from the page's own structured metadata "
        "(prefer these over the description text below if they conflict):\n"
        + "\n".join(facts)
    )
    return f"{hint_block}\n\n{text}"


def _strip_ui_noise(text: str) -> str:
    """Sprint 15.8.2: drops page-chrome lines (applicant counts,
    profile-match prompts, alert prompts, promotion labels, ...) from
    the text BEFORE it's ever put in a prompt — reuses
    rule_based._strip_ui_noise_lines()/_UI_CHROME_SENTENCE, the exact
    same filter the rule-based backend's own summary/company_overview
    derivation already applies, rather than a second noise list here.
    This is the "filtering happens before LLM extraction too" half of
    the hybrid architecture: a deterministic rule-based safety layer
    doing what it's good at (obvious pattern-matched noise removal)
    ahead of the LLM doing what it's good at (understanding what's
    left). The system prompt also asks the model to ignore this kind
    of text — this pre-filter means it doesn't have to rely on the
    model actually doing that, for the categories a fixed pattern can
    already catch outright."""
    lines = rule_based._clean_lines(text)
    return "\n".join(rule_based._strip_ui_noise_lines(lines))


class LLMExtractor(JobExtractor):
    """
    LLM-backed JobExtractor. This is a Job Understanding Layer, not just
    a title/company/location parser: employment_type, seniority,
    required/preferred skills, tech_stack, and a summary are all
    requested via structured output (see _OUTPUT_SCHEMA) and returned
    on ExtractionResult for a caller to persist however it likes.
    responsibilities/requirements are included as their own structured
    lists.
    """

    def __init__(self, model: Optional[str] = None, client: Optional[Any] = None) -> None:
        self._model = model or get_settings().llm_model
        # Tests inject a fake client here so they never need the real
        # `anthropic` package installed or a real API key.
        self._client = client

    def _get_client(self):
        return self._client if self._client is not None else _build_client()

    def extract_from_text(
        self, text: str, source_url: Optional[str] = None, *, structured_hints: Optional[dict] = None,
    ) -> ExtractionResult:
        stripped = text.strip()
        if not stripped:
            raise ValueError("raw_text is empty — nothing to extract")
        # Sprint 15.8.2: noise-stripped for the PROMPT only — Job.description
        # below (via _extract()) always gets the original, unfiltered
        # `stripped`, same "full original text kept for skill matching"
        # contract extract_from_text() has always had.
        noise_stripped = _strip_ui_noise(stripped)
        prompt_text = _with_structured_hint_context(noise_stripped, structured_hints)
        return self._extract(
            stripped, source_url=source_url, prompt_text=prompt_text, structured_hints=structured_hints,
        )

    def extract_from_url(self, url: str) -> ExtractionResult:
        # Reuse the existing fetch + HTML-parsing logic verbatim — never
        # re-implement scraping in this backend.
        html = rule_based.fetch_job_page(url)
        parser = rule_based._PageParser()
        parser.feed(html)
        page_text = (parser.text or parser.title).strip()
        if not page_text:
            raise JobFetchError(url, "unavailable page", "no usable job content found")
        # Sprint 15.8.2: same noise-stripped-prompt-only treatment as
        # extract_from_text() — Job.description still gets the original
        # page_text, unfiltered.
        return self._extract(page_text, source_url=url, prompt_text=_strip_ui_noise(page_text))

    def _extract(
        self, text: str, source_url: Optional[str], *,
        prompt_text: Optional[str] = None, structured_hints: Optional[dict] = None,
    ) -> ExtractionResult:
        # `prompt_text` (Sprint 15.7) may carry a short structured-hint
        # preamble ahead of the same `text` — sent to the model only;
        # Job.description below always uses the original `text`
        # unchanged, so a hint block never leaks into what's shown in
        # the dashboard or matched against for skills.
        data, confidence = self._call_llm(prompt_text if prompt_text is not None else text)

        # Sprint 15.8.1/15.8.3: parsed once, reused for both the
        # deterministic company resolution below and the structured-
        # metadata overlay (employment_type/posted_at/workplace_type/
        # company_url) further down — one _parse_structured_hints()
        # call, not two.
        posting, meta = rule_based._parse_structured_hints(structured_hints)

        title = _clean_str(data.get("title"), "Untitled")
        company = _clean_str(data.get("company"), "Unknown")

        # Sprint 15.8.3: deterministic company resolution ranks above
        # the model's own free-text guess — reuses the exact same
        # shared cascade RuleBasedExtractor goes through (JSON-LD >
        # structured og:site_name metadata > the company-card
        # structural signal > confident text heuristics), so both
        # backends agree on company whenever a real, positive signal
        # exists. `weak_fallback=False`: the cascade's own blind
        # "assume line 2" last resort is deliberately excluded here —
        # it has no real signal behind it, and must never override a
        # capable model's own contextual inference (see
        # _resolve_company()'s docstring). The model's own guess is
        # still the actual last resort in the overall priority order,
        # just below every tier that carries a genuine signal, never
        # below a blind guess with none.
        #
        # Sprint 15.8.4: scanned against the ORIGINAL `text`, not the
        # noise-stripped `prompt_text` — the noise filter now strips
        # company-card metric lines ("<count> followers"/"<count>
        # employees", see _UI_CHROME_SENTENCE) so the model doesn't
        # mistake them for prose, but _guess_company_from_card() needs
        # those exact lines as its anchors. Stripping them first would
        # silently blind the company-card signal on this path only —
        # RuleBasedExtractor never has this conflict since it scans the
        # unfiltered text directly.
        resolved_company = rule_based._resolve_company(
            rule_based._clean_lines(text), posting, meta, title,
            weak_fallback=False,
        )
        if resolved_company:
            company = resolved_company

        # Sprint 15.7: a defensive guard, not just a prompted rule — a
        # model doesn't always follow instructions, and this is exactly
        # the failure mode reported in production (company extraction
        # returning the job title). Cheap and deterministic: if the
        # model violated the "title and company must never match" rule
        # anyway, fall back to "Unknown" rather than trust it. Runs
        # after the deterministic override too, in case a resolved
        # company somehow still matches the title (the shared cascade's
        # own validation already guards against this, but this is the
        # last line of defense regardless of source).
        if company.strip().lower() == title.strip().lower():
            company = "Unknown"
        location = _clean_str(data.get("location"), "Unknown")
        department = _clean_optional_str(data.get("department"))
        employment_type = _clean_optional_str(data.get("employment_type"))
        seniority = _clean_optional_str(data.get("seniority"))
        summary = _clean_optional_str(data.get("summary"))
        responsibilities = _clean_str_list(data.get("responsibilities"))
        requirements = _clean_str_list(data.get("requirements"))
        required_skills = _clean_str_list(data.get("required_skills"))
        preferred_skills = _clean_str_list(data.get("preferred_skills"))
        tech_stack = _clean_str_list(data.get("tech_stack"))
        education_requirements = _clean_str_list(data.get("education_requirements"))
        company_overview = _clean_optional_str(data.get("company_overview"))

        # Sprint 15.8.2: defensive validation, not just a prompted rule —
        # the same "don't just trust the model" precedent as the
        # company==title guard above. The system prompt asks the model
        # to ignore page chrome, and the prompt text itself is already
        # noise-stripped (see _strip_ui_noise() above) — but a model can
        # still restate a noise phrase in its own words, or the noise-
        # stripping regex can miss a phrasing it's never seen. Reusing
        # the SAME rule_based._UI_CHROME_SENTENCE pattern (not a second
        # noise list) as a last check: if the model's own summary/
        # company_overview still trips it, fall back to the rule-based
        # extractor's own derivation for that one field — the hybrid
        # architecture's safety layer catching what the understanding
        # layer missed, rather than either trusting bad output or
        # leaving the field empty. Computed lazily; the common case (a
        # clean model response) never needs the fallback pass at all.
        #
        # Sprint 15.8.11 (Phase 2 Round 2): a second, independent check —
        # UI chrome and "wrong semantic category" are different failure
        # modes (a company-mission sentence isn't page noise; it's
        # legitimate prose that's simply about the wrong subject), so
        # this doesn't fold into the chrome check above. Deliberately
        # conservative (AND, not OR) to avoid rejecting a perfectly
        # good, plainly-worded summary that just doesn't happen to use
        # "you"/a title keyword: only flags summary when it BOTH (a)
        # shows no role-grounding signal at all AND (b) actively matches
        # known generic company-pitch phrasing — reuses the exact same
        # rule_based._is_role_grounded()/_looks_like_generic_company_
        # pitch() helpers _derive_summary() itself now uses, so both
        # backends agree on what "sounds like it's about the role"
        # means from one shared place, not two.
        summary_is_noisy = bool(summary and rule_based._UI_CHROME_SENTENCE.search(summary))
        overview_is_noisy = bool(company_overview and rule_based._UI_CHROME_SENTENCE.search(company_overview))
        summary_is_off_topic = bool(
            summary
            and not rule_based._is_role_grounded(summary, title)
            and rule_based._looks_like_generic_company_pitch(summary)
        )
        # Sprint 15.8.12 (Phase 2 Round 4): company_overview's own
        # off-topic check, reusing rule_based._looks_like_non_identity_
        # content() — the SAME shared definition _join_real_sentences()
        # now uses for the rule-based path's own candidate preference,
        # so both backends agree on what "not company identity" means
        # from one place. Deliberately NOT AND-gated the way summary's
        # check is: summary needed the AND because "lacks a positive
        # role signal" has real false negatives (a fine, plainly-worded
        # summary can easily use neither "you" nor a title keyword).
        # company_overview has no positive-signal-absence risk to guard
        # against here — this only fires on the ACTIVE presence of
        # specific, narrow benefits/culture/pitch/role-address vocabulary,
        # a high-precision signal on its own; requiring it to ALSO show
        # no positive signal would just make the check less sensitive
        # for no corresponding safety benefit.
        overview_is_off_topic = bool(
            company_overview and rule_based._looks_like_non_identity_content(company_overview, title)
        )
        if summary_is_noisy or overview_is_noisy or summary_is_off_topic or overview_is_off_topic:
            fallback = rule_based._extract_understanding(text, title)
            if summary_is_noisy or summary_is_off_topic:
                summary = fallback.summary
            if overview_is_noisy or overview_is_off_topic:
                company_overview = fallback.company_overview

        # Sprint 15.8.1: workplace_type/company_url/posted_at are exact
        # structured values, not something to ask the model to infer —
        # populated deterministically from the same JSON-LD this
        # backend already reuses rule_based's parser for (see
        # _with_structured_hint_context() above), never guessed by the
        # model. employment_type gets the same treatment when JSON-LD
        # has it, overriding the model's own (already-existing, Sprint
        # 9.1) free-text inference with the posting's exact stated
        # value — still falls back to the model's guess when JSON-LD
        # doesn't have it, so this is strictly additive precision, not
        # a capability regression for postings with no JSON-LD at all.
        # (`posting`/`meta` already parsed once, above, alongside the
        # company resolution.)
        posted_at = rule_based._posted_at_from_jsonld(posting) if posting else None
        company_url = rule_based._company_url_from_jsonld(posting) if posting else None
        if posting:
            employment_type = rule_based._employment_type_from_jsonld(posting) or employment_type

        # Sprint 15.8.4: workplace_type gains the same JSON-LD-priority-
        # preserved text-heuristic fallback the rule-based backend has
        # (a standalone "Remote"/"Hybrid"/"On-site" line) — this isn't
        # part of the model's own output schema at all (it's a
        # deterministic value, same reasoning as
        # workplace_type/company_url/posted_at above), so without this
        # the LLM backend would never populate it for a posting with no
        # JSON-LD, even though the raw text states it plainly.
        workplace_type = (
            (rule_based._workplace_type_from_jsonld(posting) if posting else None)
            or rule_based._guess_workplace_type_from_text(rule_based._clean_lines(text), title)
        )

        # Identity: URL is still the strongest key when known; otherwise
        # fall back to (title, company) — same rule as RuleBasedExtractor,
        # so duplicate detection behaves identically regardless of backend.
        if source_url:
            job_id = rule_based._job_id_for(source_url)
        elif title != "Untitled" or company != "Unknown":
            job_id = rule_based._job_id_for(f"{title.lower()}|{company.lower()}")
        else:
            job_id = rule_based._job_id_for(text)

        job = Job(
            job_id=job_id,
            title=title,
            company=company,
            location=location,
            url=source_url or f"local://pasted/{job_id}",
            source="linkedin" if source_url else "pasted",
            description=text,
            posted_at=posted_at,
            scraped_at=datetime.now(timezone.utc),
        )
        return ExtractionResult(
            job=job,
            title_confidence=confidence,
            department=department,
            summary=summary,
            seniority=seniority,
            employment_type=employment_type,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            tech_stack=tech_stack,
            responsibilities=responsibilities,
            requirements=requirements,
            education_requirements=education_requirements,
            workplace_type=workplace_type,
            company_url=company_url,
            company_overview=company_overview,
        )

    def _call_llm(self, text: str) -> tuple[dict, TitleConfidence]:
        """
        Call the Anthropic API and return (parsed_json, confidence).

        Never raises for a malformed/incomplete *model response* — that's
        exactly the "handle malformed JSON safely" requirement, and the
        LOW-confidence result feeds main.py's existing human-in-the-loop
        confirmation path. LLMExtractionError is reserved for the backend
        being unusable at all (no package, no credentials, request failure).
        """
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
                output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
            )
        except LLMExtractionError:
            raise
        except Exception as exc:
            raise LLMExtractionError(f"Anthropic API request failed: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            return {}, TitleConfidence.LOW

        text_blocks = [
            block.text for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        raw_text = text_blocks[0] if text_blocks else ""

        try:
            data = _extract_json_object(raw_text)
        except (json.JSONDecodeError, AttributeError):
            return {}, TitleConfidence.LOW

        if not isinstance(data, dict) or not _clean_optional_str(data.get("title")):
            return (data if isinstance(data, dict) else {}), TitleConfidence.LOW

        return data, TitleConfidence.HIGH
