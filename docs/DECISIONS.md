# Decisions

Each entry: what was decided, the evidence, and the report or run that shows it. Thresholds in
`docs/data/EDA_PLAN.md` were registered before any real sample was analysed; changing one, or the
way it is measured, is recorded here with the reason, and the original wording stays in the plan.

## D-001 (2026-10-27): criterion A7 is judged on documents with a hit, not on matches

**Plan wording:** "Personal-data hits (emails + phone-like numbers) <= 5 per 1,000 docs".

**What was measured** (fw2_arz, 47,983 train documents, first full run of the check):
853 documents contain a phone-like number or an email, but there are 3,338 phone-like matches and
212 emails, i.e. 74 matches per 1,000 documents against 17.8 documents per 1,000.

**Why the two differ:** matches pile up in a few very long pages that repeat one number. The
number `0096176817976` appears in 4 train documents of 5,500 to 8,800 words each, and the
largest pairwise similarity among them is 0.44, so they are not near-duplicates of each other.
A rate of matches therefore says more about a handful of pages than about how common personal
data is.

**Decision:** keep reporting both, and judge A7 on `docs_with_pii_per_1000` (documents with at
least one email or phone-like number per 1,000 documents). The threshold itself (5) is not
changed. Under either measure fw2_arz is far above it, so the practical consequence (a scrubbing
rule is needed for web sources) is the same; the decision only changes which number is quoted.

**Caveat:** the phone and email patterns are heuristics; a spot check of 14 fw2_arz matches found
all 14 to be real phone numbers, but wiki_ar matches were mostly ISBNs and decimals before the
pattern was tightened. Re-measure after cleaning and de-duplication, because A7 is defined on
cleaned data.
