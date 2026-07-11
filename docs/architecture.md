# Job Understanding Engine — Architecture

This document describes the **current, implemented** state of this engine — ingestion assumptions, the extraction pipeline, and the regression engineering loop that drives its quality — written so a new engineer can understand how raw posting text becomes structured data without reading the extraction module line by line.

It does not describe planned or future work — see [`design-decisions.md`](design-decisions.md) for the reasoning behind the choices made here, and this repository's README for known, documented limitations.

---

## 1. System Overview

The complete lifecycle, from raw posting text to a structured schema ready for a caller to use:

```
Raw posting text --> Extraction Pipeline --> Structured Job Schema --> (caller: matching, search, analytics, ...)
```

- **Input**: this engine takes raw text (however a caller captured it — a scrape, a paste, a stored record) and, optionally, `structured_hints` (schema.org `JobPosting` JSON-LD or Open Graph meta tags a caller already had for free).
- **Extraction Pipeline**: the raw text is turned into structured data by one of two backends (rule-based or LLM), selected behind a shared interface (§2).
- **Structured Job Schema**: ten fields with an explicit semantic contract each — not free-form text (§3).
- **Downstream**: what a caller does with `ExtractionResult` is out of this engine's scope by design — candidate matching, search indexing, analytics, anything else. This engine's contract ends at producing a correct, structured result.

**A note on the raw-input assumption baked into the design.** Several parts of this system (in particular, the fixture-export workflow described in §4) assume a caller keeps the *exact*, unmodified text it fed to an extractor available somewhere — not reconstructed, not summarized. This engine doesn't provide that storage itself (it's caller-owned), but the regression methodology depends on a caller being able to produce it, so it's worth stating as an assumption up front rather than leaving it implicit.

## 2. Extraction Pipeline

```
Raw Job Text --> ExtractorRouter --[primary]--> LLMExtractor --[title_confidence HIGH/MEDIUM]--> ExtractionResult
                        \--[fallback: LOW confidence or any LLMExtractionError]--> RuleBasedExtractor --> ExtractionResult
```

Both backends implement the same `JobExtractor` interface (`extract_from_text()`, `extract_from_url()`) — nothing downstream needs to know or care which one actually ran; a caller only ever depends on getting back "a `JobExtractor`."

| | `RuleBasedExtractor` | `LLMExtractor` |
|---|---|---|
| **Mechanism** | Regex/heuristic — explicit `Title:`/`Company:` fields, `"As a ROLE..."` sentence patterns, structured-header detection, JSON-LD parsing, generic section-header scanning for the understanding fields | Anthropic API, structured JSON output (a fixed schema, never free-form parsing) |
| **Strengths** | Zero cost, fully deterministic, no network dependency, every output traceable to an exact function/line — the property the entire regression corpus (§4) depends on | Genuine semantic understanding — handles phrasing, inline qualifiers, and structural variation no fixed pattern list can anticipate |
| **Limitations** | Can only recognize header phrasings and structures it has already been taught; zero semantic understanding (an inline "this is optional" qualifier is invisible to it); a permanently incomplete skill catalog by design | Per-request cost; output can vary run to run; a wrong output has no traceable cause beyond "the model was wrong" — no line of code to point to |
| **When it's used** | Always, as `ExtractorRouter`'s fallback; alone if a caller only ever instantiates `RuleBasedExtractor` directly | Preferred when a caller routes through `ExtractorRouter`; only trusted when `title_confidence` is `HIGH`/`MEDIUM` — a `LOW`-confidence or failed LLM call falls back to rules automatically, never raises |

**On regression validation — stated precisely, not implied:** the regression corpus (§4, §5) validates `RuleBasedExtractor` exclusively. Every fixture assertion in `tests/test_extraction_quality.py` runs against `RuleBasedExtractor().extract_from_text()` directly. `LLMExtractor` has never been exercised by this corpus — invoking the real Anthropic API from a test suite would introduce real cost and non-determinism into what's meant to be a fast, deterministic regression gate. This is a genuine, currently-open validation gap for the LLM path, not something already covered and merely undocumented — see §5 for the full statement of what is and isn't validated.

## 3. Structured Job Schema

**Why structure matters:**

```
Raw posting text                       Structured Representation
(human-oriented document —       -->   (machine-readable hiring intelligence —
inconsistent headers, UI noise,        ten fields, each with an explicit
duplicated content, ambiguous          semantic contract)
structure)
```

A raw posting is written to be read by a person scrolling a page, not queried by a program — the same information (a company's identity, a role's actual duties) can appear in a dozen different shapes across different postings. Structured extraction's job is to normalize that variation into a fixed schema any downstream feature can depend on without knowing anything about how any individual posting happened to be formatted.

| Field | Answers |
|---|---|
| `title`, `company`, `location`, `workplace_type` | Basic job identity |
| `company_overview` | "What is this organization and what does it do?" — identity only; explicitly excludes benefits, culture, DEI language, and hiring pitch, even when they appear in the same source section |
| `summary` | "What will this person in this role work on, and why does it matter?" — role-focused; explicitly excludes company mission and generic recruiting pitch |
| `responsibilities` | Itemized day-to-day duties, as discrete items, not one paragraph |
| `requirements` | Qualifications/experience/credentials the posting states it wants |
| `required_skills` / `preferred_skills` | Catalog-matched technology/tool names, explicitly named in the source text — never inferred or expanded (see [`design-decisions.md`](design-decisions.md#why-explicit-extraction-instead-of-skill-ontology-expansion)) |

Each field's contract was arrived at the same way every fix in this system was: by observing a real posting where the field's *previous*, looser definition produced the wrong content, and tightening the contract to match — see [`design-decisions.md`](design-decisions.md) for the reasoning and §4 below for the mechanism that surfaces these gaps in the first place.

## 4. Regression Engineering Loop

This is the mechanism that actually drives the quality of everything described above — not a one-time design, an ongoing loop. See [`regression-corpus.md`](regression-corpus.md) for the full standalone explanation, including documented real failure/fix examples.

```
Real usage --> Extraction failure observed --> Candidate fixture --> Audit
   ^                                                                   |
   |                                                                   v
   +---------------------------------- Regression protection <-- Root cause classification --> Targeted fix
```

- **Real usage → failure observed**: a real extraction issue is noticed while using the engine against real posting text. The exact input text the extractor saw is captured — never a hand-copied approximation.
- **Candidate fixture**: the captured text is a candidate, not yet part of the regression corpus — its `expected` block starts entirely `null`.
- **Audit**: the real extractor is run against the candidate's text, field by field, and the actual output is compared against what the posting's own text says should be correct.
- **Root cause classification**: every finding is classified precisely — is this a header-coverage gap (a pattern list missing a real phrasing), a boundary/coupling issue (does fixing one field risk leaking content into an adjacent one — always checked with a non-destructive before/after test, before any file is edited), a catalog-coverage gap, or already an existing, documented, accepted limitation? Not every observed difference becomes a fix — an issue only proceeds past this stage if it's real, reproducible, not already covered, and worth protecting against.
- **Targeted fix**: the smallest change the evidence actually supports — a single header pattern addition, a single length-cap adjustment, never a broad speculative list (see [`design-decisions.md`](design-decisions.md#why-evidence-driven-fixes-instead-of-broad-heuristics)).
- **Regression protection added**: the fixture is promoted into `tests/fixtures/`, `expected` is filled in for every field now confirmed correct (and *only* those — an unfixed field stays `null`), and at least one synthetic test using different wording is added to prove the fix generalizes rather than being fitted to the one posting that reported it.

**Why fixtures are not random examples — they represent historical engineering failures.** Every fixture in `tests/fixtures/` exists because a real posting broke something. Each carries a `provenance` block recording exactly what was found wrong, the confirmed root cause, why the fix works, and — critically — what remains open, if anything. A fixture with a `null` expected value isn't an incomplete test; it's an honest record that a specific field isn't fixed yet, deliberately left unpinned so the corpus can never overstate its own coverage. Reading the corpus is reading a record of what actually went wrong in this system and why, not a static snapshot of assumed-correct behavior.

## 5. Quality Validation

**Current validation strategy:**

- **Regression fixtures** (`tests/fixtures/`) — anonymized, derived from real extraction failures, each with a `provenance` record of its origin and an `expected` block of confirmed-correct field values.
- **Field-level expected outputs** — every fixture is checked field by field (`title`, `company`, `location`, `workplace_type`, `company_overview`, `summary`, `responsibilities`, `requirements`, `required_skills`, `preferred_skills`) against the real extractor's output; a regression in any single field fails independently, with the exact mismatch shown, not a pass/fail blob.
- **Full test suite** — `pytest tests/`, run before any change is considered complete.

**What is validated:** `RuleBasedExtractor`'s behavior against every field of every fixture in the corpus, plus every documented fix's non-regression against every *other* fixture in the corpus (confirmed directly, not assumed, at the time each fix landed).

**What is intentionally not validated yet — stated explicitly, not left implicit:**

- **LLM backend real API validation.** The corpus never invokes `LLMExtractor` against the real Anthropic API — every assertion runs against `RuleBasedExtractor` directly. This is a deliberate boundary (real API calls in a test suite mean real cost and non-determinism), not an oversight, but it is a genuine, open gap: `LLMExtractor`'s actual output quality on any of these fields, on any of these fixtures, has never been directly measured.
- **Deterministic validation for `LLMExtractor`'s `responsibilities`/`requirements`/skill fields.** `summary` and `company_overview` have a deterministic fallback check (if the model's output looks off-topic or noisy, fall back to the rule-based result) — `responsibilities`/`requirements`/skills do not. A confirmed, structural gap in the code, not yet evidenced by an observed bad output.

## 6. Architecture Decisions

The reasoning behind the choices in this document — why hybrid extraction instead of one backend, why a regression corpus instead of synthetic tests, why evidence-driven fixes instead of broad heuristics, why explicit extraction instead of skill ontology expansion — is documented separately, with real examples from this system's own history, in [`design-decisions.md`](design-decisions.md).
