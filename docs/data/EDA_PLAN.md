# EDA plan

Purpose: every analysis below exists to settle a named decision. If a result cannot change a
decision, it does not belong in this plan.

## Sampling (defaults, adjustable)

- 50,000 documents per source; 10,000 per dialect config (`acm_Arab` has only 7,382 train docs, so take all).
- Draw row groups uniformly at random across **all** shards of a source (fixed seed), then up to
  250 random rows from each. Never "first N" and never sequential streaming: the first 50k-doc
  `fw2_arz` sample came from a single-shard source read sequentially, so it is the head of that file.
  The manifest records the dataset revision, seed, sampling parameters and counts (files, row groups
  read/failed, rows); the exact row groups are re-derivable from those.
- Split off validation/test **before** any analysis that informs cleaning (see SOURCES.md).
- Memory: process in chunks; the target machine has 7 GB RAM total.

## Questions, measures, decisions

| # | Question | Measure | Decision it settles | Later consumer |
|---|----------|---------|---------------------|----------------|
| 1 | Language and script mix | Unicode-block ratios per doc; Arabic/Latin/other share; mixed-doc rate | Language ratio of the corpus; whether to filter by script | Tokenizer training mix; data mixing |
| 2 | Document length | Chars/words/tokens (reference tokenizer) quantiles | `context_length`; packing vs truncation | Data loader; model config |
| 3 | Duplication | Exact hash; MinHash near-dup rate at several thresholds | Dedup method and threshold | Corpus builder; eval hygiene |
| 4 | Arabic surface variation | Rates of diacritics, tatweel, alef/yeh/teh-marbuta variants, Arabic-Indic vs ASCII digits | Normalization rules (each rule ablated later) | Tokenizer; fertility results |
| 5 | Quality | Boilerplate, URL/number density, symbol ratio, repeated lines, `wordlist_ratio`, `language_score` | Filter thresholds | Corpus builder |
| 6 | Dialect vs MSA | Dialect-marker lexicon rates per config; label noise between `arb_Arab` and dialect configs | Whether to keep dialect labels or use them only for reporting | Eval slices; SFT data |
| 7 | Personal data | Regex rates for emails, phones, IDs | Scrubbing rules | Corpus builder; release policy |
| 8 | Token budget | Tokens per doc after cleaning, extrapolated per source | Whether available data meets ~20 tokens/param | Model size; training plan |
| 9 | Eval leakage | Overlap of eval docs with train (exact + MinHash) | Eval validity | Perplexity reporting |

## Acceptance criteria (written before any real sample was analysed)

Pre-registered on 2026-10-25, after a 200-doc smoke sample of `wiki_ary` only (used to test the
pipeline, not to set these numbers). Thresholds are starting values. Changing one after seeing
results requires a dated entry in `docs/DECISIONS.md` that states the reason, and the original
value stays visible in the report.

"Cleaned" means: exact-duplicate removal, boilerplate/footer stripping rules found in the EDA,
and a minimum-length filter. Each criterion is evaluated per source, on the train split only.

| # | Criterion | Threshold | If it fails |
|---|-----------|-----------|-------------|
| A1 | Retention: cleaned docs / sampled docs | >= 50% | Source is mostly junk; inspect by hand, drop unless a cleaning rule fixes it |
| A2 | Near-duplicate rate: share of docs with a MinHash (Jaccard >= 0.8) neighbour in the sample | <= 30% | Mandatory near-dedup; all later numbers use the deduplicated corpus |
| A3 | Language/script: Arabic-script share of letters (Arabic sources) | >= 80% | Add a script filter; recompute retention (A1) |
| A4 | Language/script: Latin share of letters (English sources) | >= 90% | Same as A3 |
| A5 | Median document length after cleaning | >= 300 chars | Add a minimum-length filter; recompute retention (A1) |
| A6 | Estimated usable tokens after cleaning and dedup (extrapolated from the sample to the available source size) | >= 20M | Source is eval/slice-only, not training data |
| A7 | Personal-data hits (emails + phone-like numbers) | <= 5 per 1,000 docs | Scrubbing rule required before the source is used |
| A8 | Train/eval exact overlap after purge | 0 docs | Do not report any perplexity until fixed |
| A9 | Train/eval near-duplicate overlap (Jaccard >= 0.8) | <= 0.5% of eval docs | Purge the matching train docs and re-measure |

A source that fails A1 or A6 and cannot be fixed by a rule is demoted to evaluation-only or dropped.
Dialect labels (question #6) get no threshold: the result is reported and decides how labels are used.

Known pitfalls to check by reading text, not only by counting:

- Wikipedia dialect editions contain templated stubs and maintenance footers (seen in the
  `wiki_ary` smoke sample); web sources will have their own boilerplate.
- Arabic diacritics: the shadda appears in dialect orthography. Do not strip diacritics blindly;
  decide per character class from the measured rates (question #4).

## Outputs

- `reports/eda/<run_id>/report.md` plus JSON of all numbers, generated by `scratchllm data analyze`.
- A one-line "decision log" entry in `docs/DECISIONS.md` per decision, citing the report run id.

## Work split

- Maintainer-written plumbing: sampling, splits, provenance, CLI, report generation, tests.
- Learner-written analysis functions: #4 (surface variation), #1 (script mix), #3 (near-duplicates).
- The rest start as the maintainer's implementation and are explained after writing.
