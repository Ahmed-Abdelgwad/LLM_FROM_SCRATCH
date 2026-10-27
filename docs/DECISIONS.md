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

## D-002 (2026-10-27): sources that cannot meet A6 are evaluation-only

**Evidence** (report `final-v1`; token counts are character-based estimates at 3.5 to 4.5
characters per token, not measured, because no tokenizer exists yet): usable tokens available
fw2_acm about 3 to 4 million, wiki_ary about 1 to 2 million, fw2_apc about 21 to 27 million before
any cleaning. Every other source is above 280 million. Even at 2 characters per token fw2_acm and
wiki_ary stay far below the 20 million of A6.

**Decision:**

- `fw2_acm` and `wiki_ary` are evaluation and slice-only. Their train split is not used for
  training; their validation and test splits are the Iraqi and Moroccan-Wikipedia evaluation slices.
- `fw2_apc` is undetermined. It is kept out of training until A6 is re-measured with the real
  tokenizer on cleaned text (cleaning will lower the 21 to 27 million), then it is judged by the
  unchanged threshold.
- Training dialects are therefore Egyptian (`fw2_arz`), Najdi (`fw2_ars`) and Moroccan web
  (`fw2_ary`), next to MSA (`fw2_arb`, `wiki_ar`) and English (`fineweb_edu_10bt`).

**Consequence to state in any result:** a model evaluated on Iraqi or Levantine text has not seen
that dialect's own data; its score measures transfer, not learning.

## D-003 (2026-10-27): web sources get a scrubbing rule; A7 is judged before and after it

**Evidence** (`final-v1`; train documents with at least one email or phone-like number per 1,000):
fw2_arb 23.3, fw2_arz 20.4, fw2_ary 5.6, fw2_ars 51.2, fw2_apc 8.0, fw2_acm 30.7,
fineweb_edu_10bt 41.6; wiki_ar 0.3 and wiki_ary 0.3 pass. Most such documents carry one to four
hits (fw2_arb 803 with one and 273 with two to four, against 49 with five or more), so they are
ordinary pages with a contact number, not spam. Pages with five or more hits are 0.1 to 2.3 per
1,000 documents and are the repeated-number spam seen in D-001.

**Decision:**

- Matched emails and phone-like numbers are replaced inside the text by the fixed placeholders
  `<email>` and `<phone>`; both are reserved as tokenizer special tokens so they cost one token.
- A document with five or more matches is dropped as spam, not masked.
- Wikipedia sources are exempt (they pass A7 as measured).
- A7 is reported twice: before scrubbing (the table above, informational) and after, as the
  number of matches a second scan still finds. After scrubbing it is zero for the patterns we
  know, so that number alone proves nothing.

**Caveat, and what the real test is:** recall is unknown. The patterns only catch the shapes in
`checks_basic.py`; names, addresses and formats we did not think of are not found. The check that
means something is reading a random sample of the masked spans (precision) and a random sample of
documents the scrubber left untouched (recall), by hand, before the corpus is built.

## D-004 (2026-10-27): mixed-script documents are a quality signal, not a script-filter target

**Evidence** (`final-v1` plus reading 6 random mixed documents each from fw2_apc and fw2_arz, a
small sample): documents with at least 20% Arabic and 20% Latin letters are 6.9% of fw2_apc, 3.9%
of fw2_arz, 2.4% of fw2_arb, 2.1% of wiki_ar, 1.4% of fw2_ars and fw2_acm, 0.6% of wiki_ary,
0.3% of fw2_ary, and one document of fineweb_edu_10bt.
The ones read were Google Books search stubs ("What people are saying - Write a review" plus a
list of letter pairs), music download pages with transliterated titles, forum posts, pirated
streaming pages and adult spam. Genuine code-switching exists too: an Egyptian Wikipedia article
quoting English band and song titles.

**Decision:**

- No filter on script share alone: it would remove the code-switching the model should learn.
- Handle the cause instead: boilerplate rules for the repeated stubs (found by reading the
  highest-frequency lines per source), and an adult-content filter, both measured for retention
  (A1) before they are accepted.
- Keep A3 and A4 as they are.

**Not yet known:** how much of the mixed share is junk versus real code-switching. Six documents
per source cannot answer that; a labelled sample of about 100 per source can, and it should be
read before the boilerplate rules are written. The adult-content filter has no method chosen yet
(URL domain list, keyword rate, or a classifier); the choice depends on the measured rate.

## D-005 (2026-10-27): templated Wikipedia stubs and short documents

**Evidence:** wiki_ar has 28.6% of documents under 300 characters, wiki_ary 20.7%, against 0.1% to
4.3% for the web sources. Near-duplicates are 2.0% of wiki_ar (325 clusters) and 1.55% of wiki_ary
(29 clusters); the largest clusters are generated village stubs ("X is a village in Y, in
province Z" with a fixed footer, 25 pages in one cluster of wiki_ar and 15 of wiki_ary). Web
sources are at 0.28% or less.

**Decision:**

- Minimum length 300 characters for every source (this is the A5 line, applied as a filter).
- Near-duplicate removal (keep one document per cluster, Jaccard >= 0.8) is applied to the
  Wikipedia sources only. A2 passes everywhere, so it is not mandatory elsewhere, and MinHash over
  the full web corpus is expensive for under 0.3% of documents.
- A1 (retention >= 50%) is expected to hold on length alone, but it is measured, not assumed,
  once the cleaning rules exist.

## D-006 (2026-10-27): Arabic normalization is decided by ablation, never by counts

**Evidence** (`final-v1`, per 1,000 Arabic letters unless stated): the diacritic and spelling rates
differ by source far more than by dialect. Shadda is 13.59 in wiki_ary (89% of all diacritics) and
0.34 to 1.83 elsewhere, so it is an editing convention of Moroccan Wikipedia, not of Moroccan
Arabic: fw2_ary (web) has 0.46. Final heh instead of teh marbuta is 40% (arz), 59% (acm), 68%
(ars), 47% (apc) against 17% in fw2_arb. Plain alef is 84% to 96% of alef forms everywhere.

**Decision:**

- Applied to every source because they lose no information a reader uses: remove the tatweel,
  and Unicode NFKC for the presentation-form blocks (copied PDF text).
- Not applied by default, and each decided by a measured ablation (tokenizer fertility and the
  language-model loss, see below): alef unification, yeh versus alef maqsura, teh marbuta versus
  heh, removal of the diacritics, mapping of digit systems.
- The shadda is never removed by a blanket diacritic rule before that ablation.
- Counts in the report justify which candidates are worth testing; they do not justify a rule.

**Open problem to settle before the ablation (D-007):** perplexity from models with different
normalizations is not comparable: the text, the tokenization and the number of tokens all change.
The comparison needs a fixed denominator (for example bits per character of one shared reference
text, with each model's own normalization applied to it) and a stated reason why that is fair.
