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

**Checked on all nine sources (2026-10-28):** the verdict is the same under both measures in every
source. Matches per 1,000 documents against documents per 1,000: fw2_arb 38.4 / 23.3, fw2_arz
74.0 / 20.4, fw2_ary 7.6 / 5.6, fw2_ars 92.5 / 51.2, fw2_apc 11.7 / 8.0, fw2_acm 67.1 / 30.7,
fineweb_edu_10bt 67.5 / 41.5 fail on both; wiki_ar 0.5 / 0.3 and wiki_ary 0.4 / 0.3 pass on both.
The new measure can never be higher than the old one (a document with a hit has at least one
match), so this is a relaxation made after seeing one source; the old figure stays in every
report as `hits_per_1000_docs`.

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

- `fw2_acm` and `wiki_ary` are evaluation-only. No part of them is used for training.
- `fw2_apc` is undetermined. It is kept out of training until A6 is re-measured with the real
  tokenizer on cleaned text (cleaning will lower the 21 to 27 million), then it is judged by the
  unchanged threshold.
- Training dialects are therefore Egyptian (`fw2_arz`), Najdi (`fw2_ars`) and Moroccan web
  (`fw2_ary`), next to MSA (`fw2_arb`, `wiki_ar`) and English (`fineweb_edu_10bt`).

**Why evaluation-only, stated correctly:** A6 says these sources are too small to train on, and
they are: `fw2_acm` holds 7,382 documents in the whole source and `wiki_ary` 8,087, so there is no
more data to draw. But A6's 20 million is a starting value that was never tested, and about 4
million tokens of `fw2_acm` is roughly 1% of the budget of a small model; it could be trained on
with repetition. The reason to keep them out is that held-out dialects give a real zero-shot
transfer measurement. That is a design choice, and it holds only if the next condition is met.

**Consequence to state in any result:** a model evaluated on Iraqi or Levantine text has not seen
that dialect's own data; its score measures transfer, not learning.

**Amended 2026-10-28:**

- The whole source is the evaluation set, not only its validation and test splits. Those splits
  are small (`fw2_acm` 88 and 471 documents, `fw2_apc` 150 and 180, `wiki_ary` 171 and 168; they
  are assigned by domain, so their sizes are uneven), and an evaluation-only source has no use for
  its train split. This gives 7,382 and 8,087 evaluation documents instead of a few hundred.
- Transfer is only zero-shot if these sources do not overlap the training sources. The overlap
  (same crawl, same sites, near-duplicate pages) has not been measured; it is checked with the
  A8 and A9 procedure across sources before any dialect result is reported.
- Dialect labels come from a language-identification model and are approximate: `fw2_ary` web
  text resembles MSA in `final-v1` (final heh instead of teh marbuta 14.4% against 17.4% in
  fw2_arb, plain alef 87% against 84%). A slice called "Iraqi" means "labelled Iraqi by that model".

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

**Evidence** (`final-v1`): documents with at least 20% Arabic and 20% Latin letters are 6.9% of
fw2_apc, 3.9% of fw2_arz, 2.4% of fw2_arb, 2.1% of wiki_ar, 1.4% of fw2_ars and fw2_acm, 0.6% of
wiki_ary, 0.3% of fw2_ary, and one document of fineweb_edu_10bt.

**Measured on the whole train sample, then checked by reading** (60 random mixed documents each
from fw2_apc, fw2_arz, fw2_arb, wiki_ar, fw2_ars and fw2_acm, drawn with `scratchllm data read`):

- Google Books search stubs are a fw2_apc problem: 382 of 9,670 train documents (3.95%) contain
  the stub text ("What people are saying - Write a review", "Try this search over all
  volumes", "We haven't found any reviews in the usual places"). They are 28 of 47,939 in
  fineweb_edu_10bt and 0 in every other source. 292 of fw2_apc's 667 mixed documents are on
  google.com.
- The rule must test the text, not the domain: 107 other google.com documents in fw2_apc are
  `ejabat.google.com` question-and-answer threads in Levantine Arabic, which are real text.
- Mixed is not junk by itself. 262 of fw2_arb's 1,160 mixed documents are tripadvisor.com.eg
  pages (hotel reviews and listings with English place and hotel names), and all 1,027 mixed
  wiki_ar documents are Wikipedia articles with foreign names, formulas and "(in English: ...)"
  glosses. Both are text the model should read.
- Junk in the mixed share, from the reading: music-download pages with transliterated titles,
  streaming and film-listing pages, keyword-stuffed pages on generated hosts (`.web.app`,
  `.ru/post/`), and adult spam. These share no single text signature yet.

**Decision:**

- No filter on script share alone: it would remove the code-switching the model should learn.
- Handle the cause instead. The first boilerplate rule is the Google Books stub signature:
  documents containing any of the three stub sentences are dropped (about 4% of fw2_apc). No rule
  is written for the other junk until its share is measured; an adult-content filter is also
  needed. Each rule is measured for retention (A1) before it is accepted.
- Keep A3 and A4 as they are.

**Not yet known:** the junk share of the remaining mixed documents (60 per source gives an
impression, not a rate) and the method of the adult-content filter (URL domain list, keyword rate,
or a classifier), which depends on the measured rate. A labelled sample of about 100 documents per
source, written down document by document, would give the rate.

## D-005 (2026-10-27, amended 2026-10-28): templated Wikipedia stubs, short documents, leakage

**Evidence:** wiki_ar has 28.6% of documents under 300 characters, wiki_ary 20.7%, against 0.1% to
4.3% for the web sources. Near-duplicates are 2.0% of wiki_ar (325 clusters) and 1.55% of wiki_ary
(29 clusters); the largest clusters are generated village stubs ("X is a village in Y, in
province Z" with a fixed footer, 25 pages in one cluster of wiki_ar and 15 of wiki_ary). Web
sources are at 0.28% or less.

**Read on 2026-10-28** (14 random short train documents of wiki_ar, 8 of wiki_ary):

- wiki_ar short documents are one defining sentence, a "references" heading and a list of category
  names. They carry little information, but they are true text, not junk.
- The lengths above include that category list, so a 400-character document can be 150 characters
  of prose. The 300-character threshold and the 28.6% / 20.7% were measured on the wrong quantity.
- wiki_ary short documents include real Moroccan Arabic prose (Robert Boyle, the airship,
  Borobudur) next to generated day-of-year pages and village stubs. 20.7% is not junk.

**Leakage measured** (MinHash, Jaccard >= 0.8, all splits of the sample): 38 of wiki_ar's 2,013
evaluation documents (1.9%) have a near-duplicate in the train split, against the 0.5% limit of
A9; wiki_ary has 2 of 339 (0.6%, too few to read as a rate). The samples are 50,000 and 8,000
documents out of millions, so a full corpus has more neighbours per evaluation document and the
real rate is likely higher.

**Decision (replaces the first version, which set a 300-character filter for every source and
near-duplicate removal for the Wikipedia sources):**

- Removing train documents that are near-duplicates of an evaluation document (A9) is mandatory,
  for every source, and measured across sources too. This is the reason to de-duplicate, more than
  the 2% that A2 counts.
- Near-duplicate removal inside the train split (keep one document per cluster) is optional,
  applied to the Wikipedia sources only, and decided after the leakage purge shows how much is
  left. A2 passes everywhere.
- The category-list footer is stripped from Wikipedia documents first. A minimum length is then
  measured on the prose that remains, from its distribution, and its threshold is chosen then;
  300 stays only as the A5 reference for the median.
- No length filter on evaluation-only sources (D-002): filtering them would shift the evaluation
  towards longer articles.
- A1 (retention >= 50%) is not assumed to hold; it is measured once the rules exist.

## D-006 (2026-10-27, amended 2026-10-28): lossy Arabic normalization is a requirement, not a tuned setting

**Evidence** (`final-v1`, per 1,000 Arabic letters unless stated): the diacritic and spelling rates
differ by source far more than by dialect. Shadda is 13.59 in wiki_ary (89% of all diacritics) and
0.34 to 1.83 elsewhere, so it is an editing convention of Moroccan Wikipedia, not of Moroccan
Arabic: fw2_ary (web) has 0.46. Final heh instead of teh marbuta is 40% (arz), 59% (acm), 68%
(ars), 47% (apc) against 17% in fw2_arb. Plain alef is 84% to 96% of alef forms everywhere.

**Decision:**

- Applied to every source because they lose no information a reader uses: remove the tatweel,
  and Unicode NFKC for the presentation-form blocks (copied PDF text).
- Not applied: alef unification, yeh versus alef maqsura, teh marbuta versus heh. These change
  words ("على" and "علي"), so a generative model trained with them cannot write correct spelling.
  Keeping them is a requirement of the model, not a setting to be tuned by loss.
- Open, not decided: removal of the diacritics and mapping of the digit systems. They are lossy
  but rarely change a word; decide them with the tokenizer in hand.
- The shadda is never removed by a blanket diacritic rule.
- Counts in the report justify which candidates are worth discussing; they do not justify a rule.

**Amended 2026-10-28:** the first version said each lossy rule would be settled by an ablation on
fertility and loss, and proposed comparing models by bits per character of a shared text with each
model's own normalization applied. That comparison is wrong, see D-007, so lossy rules are decided
by what the model has to be able to write.

## D-007 (2026-10-28): models with different lossy normalization cannot be compared by likelihood

**Problem:** a lossy normalization merges different spellings into one, so the normalized text has
less entropy than the original. A model trained on it gets a lower bits-per-character on its own
normalized text even if it is the worse model, so scoring each model on its own normalization is
not a comparison. Scoring all models on the text normalized the strongest way favours the model
trained that way. Scoring on the raw text is impossible for a model that cannot tell the variants
apart. Perplexity per token is also not comparable across tokenizers.

**Decision:**

- No likelihood comparison between models or tokenizers trained on differently normalized text.
- A comparison across normalizations needs a task whose inputs are fixed raw text and whose score
  does not depend on the normalization (for example dialect identification), and a stated reason
  why it measures what we care about. None is chosen yet.
- Tokenizer fertility is reported for every normalization tried, labelled as a description of the
  tokenizer on its own normalized text, not as evidence that one normalization is better.
