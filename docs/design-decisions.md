# Design Decision Records

This document explains *why* the job understanding pipeline is built the way it is — not what the code does (the code and [`architecture.md`](architecture.md) already say that), but the reasoning behind the decisions that would otherwise look arbitrary from the outside. Every example below is a real change made in this project's history, not a hypothetical.

---

## Why hybrid extraction?

**Decision:** extraction runs behind a `JobExtractor` interface with two real implementations — `RuleBasedExtractor` (deterministic, zero-dependency, no network call) and `LLMExtractor` (Anthropic-backed, structured output) — plus `ExtractorRouter`, which tries the LLM first and falls back to rules on low confidence or any failure.

**Why not LLM-only:**
- **Cost.** Every extraction is a paid API call. A test suite, a CI pipeline, or a bulk re-import all multiply that cost by however many postings are involved.
- **Determinism.** The same input can produce a different output from run to run. A regression suite needs a stable baseline to assert against — `RuleBasedExtractor` is the layer that provides one.
- **Debuggability.** When `RuleBasedExtractor` produces a wrong output, the root cause is traceable to an exact function and line (a header pattern, a length cap, a content filter). When an LLM produces a wrong output, there is nothing to trace — only the input and output are observable, not the reasoning in between.
- **Availability.** No API key, no installed `anthropic` package, and no network access are all failure modes the rule-based path is immune to by construction, not by error handling.

**Why not rule-based only — this is not a claim that the LLM is unnecessary:**
- A fixed pattern list can only recognize header phrasings it has already been taught. Every real header-coverage fix in this project's history exists *because* a real posting used phrasing the existing pattern list didn't cover (`"What The Role Involves"`, `"This Is An Opportunity To"`, `"What You Will Bring"`, and others) — this is an inherent, permanent property of pattern matching, not a bug being incrementally closed.
- Rule-based extraction has no semantic understanding at all. An inline qualifier like *"Master's a plus"* sitting inside an otherwise-required bullet is invisible to it — the mechanism can only tell "is this line under a recognized header," never "does this specific clause mean something different from the sentence around it."
- The LLM path's prompt (`src/job_understanding/llm.py`) encodes genuine semantic contracts a regex cannot — e.g., an explicit instruction that `company_overview` must never be inferred from the model's own outside knowledge, only from what the posting itself states.

**The actual tradeoff, stated plainly:** rule-based extraction is a safety layer — cheap, deterministic, debuggable, and permanently incomplete against novel phrasing. The LLM is the layer capable of genuine understanding — and permanently more expensive, non-deterministic, and harder to root-cause when wrong. `ExtractorRouter` doesn't resolve this tradeoff; it makes a specific, explicit choice about it: prefer the LLM's understanding when it's available and trustworthy (`title_confidence` isn't `LOW`), fall back to the deterministic floor when it isn't. Neither backend is presented as sufficient alone.

---

## Why a regression corpus?

**Decision:** every confirmed extraction bug gets a permanent fixture in `tests/fixtures/`, derived from the actual raw text a real posting produced (anonymized — see [`regression-corpus.md`](regression-corpus.md) for exactly what that means) — never a hand-written approximation invented from scratch.

**What it is not:** a directory of example inputs picked because they seemed interesting. Every fixture exists because a real posting broke something, and stays permanently because the fix needs a durable guard against reintroduction.

**The workflow, as it actually operates (not aspirational — this is the process every fixture in the corpus went through):**

```
Real usage
    |
    v
Extraction failure observed
    |
    v
Candidate fixture created      the exact raw text an extractor actually
    |                           consumed for that posting — never a
    |                           hand-copied approximation
    v
Root cause analysis            traced against the actual code path: which
    |                           function, which condition, why — confirmed
    |                           by tracing, not assumed by pattern-matching
    |                           against a previous, similar-looking bug
    v
Fix implemented                the smallest change the evidence supports;
    |                           coupling risk (does fixing field A risk
    |                           leaking content into field B?) verified
    |                           with a non-destructive before/after test
    |                           BEFORE any file is edited, not after
    v
Regression protection added    a permanent fixture assertion, plus at
                                 least one synthetic test using different
                                 wording, proving the fix generalizes
                                 rather than being fitted to one posting
```

**A concrete illustration.** A real production posting (anonymized as `company_overview_truncation.json` in this corpus) surfaced three unrelated failures simultaneously: a garbled `company_overview` (two compounding causes — a length cap silently discarding the real content before a classification filter ever saw it, and a standalone UI truncation marker surviving as a bare artifact), and two empty structured fields (`responsibilities`, `requirements`) caused by header phrasings the pattern lists had never seen. Each was fixed in its own change, each verified against every *other* fixture in the corpus to confirm zero side effects, and the fixture's `expected` block only asserts the fields actually confirmed correct at each point — an unfixed field is left `null` rather than silently pinned as "correct," so the corpus can never overstate its own coverage.

**Why this, instead of a synthetic test suite:** synthetic fixtures are, by construction, shaped around what the author already expects the extractor to handle. A synthetic fixture using the canonical header `"Responsibilities"` will always pass — it can't reveal that a real posting phrases it `"This Is An Opportunity To"` instead, because nobody writing a synthetic case would think to guess that specific, real-world phrasing. Only a genuinely captured posting can expose a gap nobody anticipated. That's exactly why this corpus is derived from real failures rather than invented outright, even though the fixtures themselves are anonymized.

**The corpus preserves engineering history, not just current behavior.** Every fixture's `provenance` block records where it came from, what was found wrong, the confirmed root cause, and what — if anything — remains open. Reading the corpus is reading a record of what actually broke and why, not a static snapshot of "current expected output."

---

## Why evidence-driven fixes instead of broad heuristics?

**Decision:** every pattern-list addition, every regex widening, every new marker is added because a specific, real posting demonstrated it's needed — never speculatively, never as a "let's also cover this while we're here."

**A direct example of the discipline this enforces.** When a real posting's responsibilities header (`"This Is An Opportunity To"`) was found unrecognized, the fix wasn't "add a broad set of plausible synonyms." It was one line: that exact phrase. A later validation round explicitly checked for further candidate phrasings (`"Your Profile"`, `"About You"`, `"Must Have"`) that *looked* plausible, found zero evidence any of them appear in any real fixture or production trace, and left every one of them unadded — recorded as a watch-list, not implemented ahead of evidence.

**Why this matters more than it might first appear:** a broad, speculative pattern list is untestable in the way that matters — there's no real input to verify it against, no signal for whether a given addition helps or introduces an unintended false positive. Every addition in this project's history was verified two ways before landing: does the exact real text now match, and does a non-destructive check against every *other* fixture confirm nothing else changed. Neither check is possible for a pattern invented without a real example to test it against.

**The corollary, also enforced consistently:** a length cap or field boundary is never re-tuned to "probably cover more cases" — when a real, unusually long company-identity paragraph was found silently discarded by a too-small cap, the fix was a cap increase justified by that specific measured length (verified against the field's own existing output budget elsewhere in the code, not invented), not a round number picked because it "felt safer."

---

## Why explicit extraction instead of skill ontology expansion?

**Decision:** `required_skills`/`preferred_skills` extract only literal, explicitly-named technology strings already present in the source text, matched against a small, deliberately hand-curated catalog (`TRACKED_SKILLS`) — never inferred, expanded, or generalized into a broader concept.

**What this explicitly excludes, by design:**
- `"AWS"` mentioned in a posting extracts as `AWS` — it never expands into `EC2`/`Lambda`/`SageMaker` just because those are AWS services, even after those specific service names were separately added to the catalog as their own, independently-evidenced entries.
- `"Build AI recommendation systems"` does not infer `Machine Learning`/`Deep Learning`/`Recommendation Systems` as skills — those are domain descriptions, not named tools, and the project's catalog-matching mechanism has no inference capability at all: it can only find literal substrings already present in the text.
- `"LLM"` extracted from a posting stays `LLM` — it does not expand into `Transformers`/`RAG`/`Vector Database`, related concepts a genuine ontology might associate with it.

**Why:** this project's scope is *job description understanding and structured extraction* — turning what a posting already says into structured data. It is deliberately not *skill ontology construction*, *skill hierarchy inference*, or *skill recommendation* — each of those is a different, larger problem with a different validation story (an ontology needs to be evaluated against domain-expert judgment of what *should* relate to what; explicit extraction only needs to be evaluated against whether the literal text was captured correctly, which a regression corpus can actually verify).

**A structural guarantee, not just a policy.** Because the matching mechanism (`src/job_understanding/_skills.py`'s `build_skill_engine()`) is a pure literal-keyword regex with zero generative capability, it is *structurally* impossible for it to expand a mention into related concepts — there's no code path that could produce that behavior even by accident. This was confirmed directly, not just asserted: a real posting's "computer vision"/"3D data"/"AI/ML" domain description correctly produces zero inferred skills, not because a filter blocks the inference, but because the mechanism was never capable of inferring in the first place. Any future move toward fuzzy matching, embeddings, or LLM-assisted skill extraction would need to *actively re-establish* this guarantee — it is currently a side effect of the mechanism's simplicity, not a safeguard that was explicitly designed in.

**Catalog growth follows the same evidence discipline as every other fix.** `TRACKED_SKILLS` has been widened only when a real posting explicitly named a technology the catalog didn't yet track (e.g. `XGBoost`, `MLflow`, `ZenML`, `Metaflow`, `EC2`, `EKS`, `CloudFormation`, `Cognito`, `LLM`) — confirmed present in real posting text before being added, never added speculatively. The catalog is understood to be permanently incomplete against the full space of real technology names by design — that incompleteness is the accepted cost of a small, auditable, explicit list over an open-ended or inferred one.
