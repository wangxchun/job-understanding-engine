# The Regression Corpus

`tests/fixtures/` — why it exists, how it works, and what it actually is (and isn't).

---

## Problem

Traditional extraction systems regress silently. A rule-based extractor is a growing collection of pattern matches — header lists, length caps, content filters — and every one of those patterns can conflict with another in ways that are invisible until a specific real input hits the conflict. A fix for one posting's problem can quietly break a different posting's previously-correct output, and without something checking every previously-correct case on every change, that regression ships unnoticed.

Synthetic test cases don't solve this. A hand-written test fixture is, by construction, shaped around what its author already expects the system to handle — it can prove a known pattern still works, but it can't reveal a real-world phrasing nobody thought to write a test for. Every fix documented in this corpus exists specifically because a synthetic test would not have caught it.

## Approach

Real extraction failures become permanent regression fixtures — anonymized before inclusion here. When a real posting exposes a genuine extraction failure, the structural shape that caused it — the exact header phrasing, the exact noise pattern, the exact boundary condition — is preserved, while everything that identifies the source (company name, job ID, URL, recruiter/contact information) is stripped out. What matters for regression protection is the *failure pattern*, not who posted the job.

**Provenance policy, stated plainly:** every fixture in this corpus honestly states where it came from. A fixture derived from a real production failure says exactly that — `"source": "anonymized_real_case"`, with a note like *"Derived from a real production extraction failure; identifying details removed."* No fixture in this corpus claims to be synthetic-from-scratch when it isn't; that distinction matters because a synthetic fixture and an anonymized-real fixture prove different things (see "Why this, instead of a synthetic test suite" in [`design-decisions.md`](design-decisions.md)).

## Workflow

```
Real usage --> Extraction failure noticed --> Candidate captured --> Audit
                                                                        |
                                                                        v
                        Anonymize (strip identifying details,   <--  Confirmed failure?
                        keep structural shape) --> Permanent            |
                        regression fixture                              v (no)
                                                                  Left as candidate
                                                                  or discarded
```

- **Real usage → failure noticed.** A real extraction problem is noticed while running the engine against real posting text — not sought out, not constructed.
- **Candidate captured.** The exact text the extractor consumed for that posting is captured, unmodified.
- **Audit.** The real extractor is run against the candidate's text, field by field, and compared against what the posting's own content says should be correct. Not every observed difference is a bug — some are already-documented, accepted limitations (see "What the corpus is not," below).
- **Anonymize.** Company names, job IDs, URLs, recruiter/contact names, and any other identifying detail are replaced with generic placeholders. City/region-level location is kept where it's structurally relevant to the failure (e.g. distinguishing a location line from a title line); anything more specific is removed. The header phrasing, noise pattern, and boundary condition that actually caused the failure are preserved verbatim — anonymizing changes *who*, never *what broke*.
- **Confirmed failure → permanent regression fixture.** Only issues that are real, reproducible, not already covered by an existing fixture, and worth protecting against get promoted. The fixture's `expected` block only records fields *confirmed* correct — an unfixed field is left `null` rather than pinned as "correct" by accident, so the corpus can never overstate its own coverage.
- **Future regression protection.** Once promoted, the fixture is asserted on every test run, indefinitely. A fix for a *different* field or a *different* fixture that accidentally breaks this one is caught immediately, not discovered later.

## Examples

Three real, documented cases — anonymized, but structurally exactly what happened:

**`company_overview_truncation.json` — company overview failure.** A real captured posting's `company_overview` field resolved to garbled truncation artifacts and an unrelated company-culture sentence instead of the company's actual identity description. Root cause: two compounding issues — a length cap silently discarding the genuine, much-longer identity paragraph before any content filtering ever ran on it, and a standalone UI truncation marker (`"… more"` on its own line) surviving as a bare artifact because the existing cleanup logic was only ever designed for a marker glued to real text on the same line. Fixed by making the collection length cap field-specific (company overview text is real prose, not a short bullet, and needed a larger budget) and recognizing a standalone marker as pure noise — verified against every other fixture in the corpus to confirm zero side effects elsewhere.

**`responsibilities_header_gap.json` — responsibilities header failure.** The same underlying posting's `responsibilities` field came back completely empty, despite the posting containing a clean, itemized duty list — under a header phrasing the extractor's pattern list had never seen. Fixed by adding that exact phrase as a recognized header — and, because widening one section's header list can let content bleed into the next section if the *following* header also isn't recognized, this was verified with a non-destructive before/after test *before* any file was changed, confirming exactly what would and wouldn't leak.

**`requirements_boundary_leak.json` — requirements boundary failure.** After the responsibilities fix above, `requirements` began collecting real content correctly — but never stopped cleanly, swallowing the posting's own EEO/accommodation trailing sub-header as a bogus final item. A distinct failure from the header-recognition issue above: not a missing *start* boundary, but a missing *stop* boundary. Fixed by recognizing that specific trailing header as a stop marker — confirmed, again with a before/after test, that this had no effect on any other field in any other fixture.

Each of these is now a permanent fixture in the corpus, asserting the fixed fields exactly, and each fix shipped with at least one additional synthetic test using different wording, proving the fix generalizes rather than being narrowly fitted to the one posting that reported it.

## What the corpus is not

**It is not a benchmark dataset.** A benchmark answers "how good is this system, as a percentage?" This corpus doesn't produce a score, and one shouldn't be inferred from it — a handful of fixtures is not a statistically meaningful sample of the full variety of real job postings, and treating it as one would overstate what it actually proves.

**It is an evidence-driven regression protection system.** Its actual claim is much narrower and much more useful: *these specific real failure patterns, once broken, are now permanently verified correct, and stay that way as the system continues to change.* That's a guarantee a benchmark score doesn't give — a 95% benchmark accuracy says nothing about whether the next unrelated change breaks the one case that used to work. This corpus is built specifically to catch exactly that.

**It grows only from real evidence, never speculatively.** No fixture in this corpus was constructed to test a hypothetical edge case. Every one exists because a real posting broke something real, and every fix a fixture protects was scoped to exactly what that evidence supported — see [`design-decisions.md`](design-decisions.md#why-evidence-driven-fixes-instead-of-broad-heuristics) for why that discipline is treated as a hard constraint, not a preference.

**It is anonymized, not fabricated.** Anonymization removes *who* (company, job ID, URL, contact details); it never changes *what broke* (the header phrasing, the noise pattern, the boundary condition). A fixture's `provenance.notes` says so explicitly rather than letting a reader assume the data is either fully real or fully invented.

## Where this fits

Full extraction pipeline design: [`architecture.md`](architecture.md) (see "Regression Engineering Loop" for the mechanism itself in the context of the full system). Reasoning behind the underlying engineering choices: [`design-decisions.md`](design-decisions.md).
