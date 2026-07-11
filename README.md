# Job Understanding Engine

Evidence-driven job understanding engine that transforms noisy job postings into structured hiring intelligence.

Real job postings are noisy and inconsistent — the same information (title, company, responsibilities, requirements) is scattered across page chrome, truncated widgets, and a dozen different header phrasings, none of them standardized. This project is a **job understanding pipeline** that turns raw, messy posting text into a structured schema — and, the part that actually makes it reliable, a **regression corpus built from real extraction failures**, not synthetic test cases, so quality improves the same way a production system's quality actually improves: by catching what real input breaks, fixing the root cause, and permanently protecting against it.

This README documents the engineering system, not just the feature list.

---

## Quick Start

```bash
git clone https://github.com/wangxchun/job-understanding-engine.git
cd job-understanding-engine

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -e ".[dev]"        # dev extra pulls in pytest
pytest                         # run the full test suite

python examples/extract_job.py examples/sample_posting.txt
```

The core package (`RuleBasedExtractor`) is pure stdlib and works out of the box. `LLMExtractor` additionally needs the Anthropic SDK and an API key — see [Development](#6-development).

---

## 1. The Problem

Job descriptions are written for humans, not machines. A real captured posting mixes the actual content with:

- **Inconsistent section headers.** "Responsibilities" in one posting, "What The Role Involves" in another, "This Is An Opportunity To" in a third — all real, all observed in this project's own regression corpus, none of them a standard.
- **Duplicated and repeated company information.** A real posting behind one of this corpus's fixtures repeats the company's mission statement near-verbatim multiple times before the actual role description starts.
- **UI noise mixed into the content.** Follower counts, "Easy Apply" labels, "People also viewed" widgets, analytics charts, and — concretely — literal `"… more"` truncation-link artifacts that a naive text extraction captures as if they were real content.
- **Missing or ambiguous structure.** Some postings have no explicit "Requirements" section at all; the qualifications are folded into a paragraph. Some mark a qualification as optional inline ("Master's a plus") rather than under a dedicated "Nice to have" header.
- **Trailing boilerplate with no clear boundary.** EEO/accommodation statements, cookie notices, and "Additional Details" sections that run directly into the last real content item with no structural marker separating them.

**Why naive extraction fails:** a purely rule-based parser needs to recognize every real-world header phrasing in advance — an open-ended, permanently incomplete list. A purely LLM-based parser has no deterministic safety net, no debuggability into *why* it produced a given output, and a per-request cost. This project's answer is neither alone — see [Why hybrid extraction?](#4-key-engineering-decisions) below.

---

## 2. What This Is

A hybrid (rule-based + LLM) extraction engine that converts raw job-posting text into a structured schema: title, company, location, workplace type, company overview, summary, responsibilities, requirements, and required/preferred skills.

The engineering story worth reading this README for isn't "I wrote parsers." It's: **an evidence-driven job understanding system that continuously improves from real-world extraction failures**, with a permanent, growing regression corpus as the proof.

Three things this project emphasizes over a typical rules-plus-LLM-call script:
- **Real-world-derived data, not invented examples.** The regression corpus (`tests/fixtures/`) is derived from actual extraction failures encountered against real postings — anonymized, but structurally real (see [`docs/regression-corpus.md`](docs/regression-corpus.md) for exactly what "derived" means here).
- **Structured extraction with an explicit contract.** Every field has a defined semantic meaning (what belongs, what doesn't) — not "whatever the LLM felt like returning."
- **Regression-driven improvement.** Every fix in this project's history follows the same loop: observe a real failure → root-cause it against the actual code path → fix the smallest thing that explains the evidence → add a permanent regression fixture. No speculative fixes for problems that haven't been observed.

---

## 3. Architecture

```
        text (or URL)
              |
              v
       ExtractorRouter
              |
       ---------------
       |             |
  LLMExtractor   RuleBasedExtractor
  (primary)      (fallback: LOW confidence
       |          or any LLMExtractionError)
       -------+-------
              v
       ExtractionResult
       title · company · location · workplace_type ·
       company_overview · summary · responsibilities ·
       requirements · required_skills · preferred_skills
```

- **`RuleBasedExtractor`** (`src/job_understanding/rule_based.py`) — deterministic, zero network call, zero cost. Regex/heuristic: explicit `Title:`/`Company:` fields, `"As a ROLE..."` sentence patterns, structured-header detection, JSON-LD parsing, generic section-header scanning for the understanding fields.
- **`LLMExtractor`** (`src/job_understanding/llm.py`) — Anthropic-backed, structured JSON output (a fixed schema, never free-form parsing). Genuine semantic understanding of ambiguous phrasing, inline qualifiers, and structural variation no fixed pattern list can anticipate.
- **`ExtractorRouter`** (`src/job_understanding/router.py`) — tries the LLM first, falls back to the rule-based extractor on `LOW` title confidence or any `LLMExtractionError`. Both extractors implement the same `JobExtractor` interface (`src/job_understanding/base.py`), so a caller never needs to know which one actually produced a given result.

Full component-level detail and design principles: [`docs/architecture.md`](docs/architecture.md).

---

## 4. Key Engineering Decisions

### Why hybrid extraction?

Neither backend alone is the right answer, and this project doesn't pretend otherwise:

| | `RuleBasedExtractor` | `LLMExtractor` |
|---|---|---|
| Cost | Zero — no network call | Per-request Anthropic API cost |
| Determinism | Fully deterministic — same input, same output, every time | Model output can vary run to run |
| Debuggability | Every decision traces to a specific regex/function — a wrong output can be root-caused to an exact line | Opaque — a wrong output can only be observed, not traced |
| Semantic understanding | None — purely structural (header matching, positional heuristics) | Genuine understanding of ambiguous phrasing, inline qualifiers, novel structures |
| Coverage | Limited to header phrasings it's been taught | Generalizes to phrasing it's never seen |

**The tradeoff is real, not a marketing claim**: rule-based extraction cannot understand an inline "this is optional" qualifier the way an LLM can, and no fixed pattern list will ever cover every real header phrasing a job posting might use. But an LLM-only system has no deterministic safety net when the model degrades, times out, or is simply wrong, and every failure costs money to even observe. `ExtractorRouter` is the explicit compromise: try the LLM, fall back to rules on low confidence or any failure — the LLM's understanding when it's available, a deterministic floor when it isn't.

### Why a regression corpus?

This is the project's central engineering practice, and it's not a collection of random test cases. It represents **an evidence-driven evaluation loop built from real extraction failures** — see [`docs/regression-corpus.md`](docs/regression-corpus.md) for the full explanation.

```
Real extraction failure observed
        |
        v
Fixture created            (the exact structural shape that broke —
        |                    header phrasing, noise pattern, boundary —
        |                    anonymized: no company names, job IDs, or
        |                    other identifying details)
        v
Root cause analysis        (traced against the actual code path —
        |                    which function, which line, why)
        v
Fix implemented             (the smallest change the evidence
        |                    actually supports — no speculative
        |                    header lists, no untested heuristics)
        v
Regression protection added (a permanent fixture asserting the
                             fixed field, plus a test proving the
                             fix generalizes beyond the one case)
```

Every fixture in `tests/fixtures/` is honestly labeled: derived from a real production extraction failure, with identifying details removed — never presented as synthetic-from-scratch. See [`docs/regression-corpus.md`](docs/regression-corpus.md) for the full policy and worked examples.

More engineering decisions — why evidence-driven fixes instead of broad heuristics, why explicit skill extraction instead of ontology expansion — are documented in [`docs/design-decisions.md`](docs/design-decisions.md).

---

## 5. Project Structure

```
src/job_understanding/
  base.py               JobExtractor interface, ExtractionResult, TitleConfidence
  schema.py              Job dataclass
  rule_based.py           RuleBasedExtractor — deterministic extraction
  llm.py                   LLMExtractor — Anthropic-backed structured extraction
  router.py                 ExtractorRouter — LLM-first, rule-based fallback
  catalog.py                 The tracked-skills catalog (evidence-driven, small, editable)
  skill_normalizer.py         Deduplicates/categorizes raw skill output
  _skills.py                    Keyword-to-regex matching engine
  _config.py                     Minimal settings (API key, model name)
tests/
  fixtures/                The regression corpus — anonymized, real extraction failures
  test_extraction_quality.py   Extraction correctness + regression assertions
docs/
  architecture.md         Full extraction pipeline design
  design-decisions.md     Why hybrid extraction, why evidence-driven fixes
  regression-corpus.md    Why the regression corpus exists, how it works, real examples
examples/
  extract_job.py          Minimal usage example
```

---

## 6. Development

### Install

Installs the `dev` extra, which includes `pytest`:

```bash
pip install -e ".[dev]"
```

Add the `llm` extra too if you'll also use `LLMExtractor`:

```bash
pip install -e ".[dev,llm]"
```

### Run the tests

```bash
pytest tests/
```

Run just the extraction-quality/regression-corpus tests:

```bash
pytest tests/test_extraction_quality.py -v
```

### Try it on a job posting

Use the included example:

```bash
python examples/extract_job.py examples/sample_posting.txt
```

Or provide your own job posting text file:

```bash
python examples/extract_job.py path/to/your_job_posting.txt
```

The input file should contain raw job posting text in plain text format (for example, text copied from a job board).

```python
from job_understanding import RuleBasedExtractor

extractor = RuleBasedExtractor()
result = extractor.extract_from_text(open("posting.txt").read())

print(result.job.title, "@", result.job.company)
print(result.summary)
print(result.required_skills)
```

To use `LLMExtractor` or `ExtractorRouter`, set `ANTHROPIC_API_KEY` in your environment.

---

## 7. Real-World Validation

This isn't validated against invented examples — every fixture in the regression corpus traces back to a real extraction failure, with identifying details removed but the actual structural challenge preserved verbatim.

- **Field-level comparison, not a pass/fail blob.** Every fixture's `expected` block is checked field by field against the real extractor's output — a regression in any one field fails independently, with the exact mismatch shown.
- **No fabricated accuracy percentages or benchmark numbers are claimed anywhere in this project** — only counts directly verifiable by running the test suite. See [`docs/regression-corpus.md`](docs/regression-corpus.md#what-the-corpus-is-not) for why this project doesn't claim to be a benchmark.

---

## 8. Future Improvements

Documented, evidence-tracked limitations — not unfinished features presented as current capability:

- **LLM backend validation.** The regression corpus exercises `RuleBasedExtractor` exclusively (by design — invoking the real LLM in a test suite risks real API cost and non-determinism). The `LLMExtractor` path has no deterministic validation for `responsibilities`/`requirements`/skills the way `summary`/`company_overview` do.
- **Broader skill catalog coverage.** `TRACKED_SKILLS` is a small, deliberately hand-picked, evidence-driven list — widened only when a real posting demonstrates a missing entry, never speculatively.
- **Additional fixtures.** The corpus grows only from genuine extraction failures encountered during real usage, never invented cases — so its coverage is inherently incremental, not exhaustive.

---

## License

MIT License
