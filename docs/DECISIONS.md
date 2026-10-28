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

**Measured after the footer rule** (`scratchllm data clean-report`, rules in
`src/scratchllm/data/cleaning.py`, train split of the samples):

| | median chars before -> after | documents under 300 chars | documents >= 300 chars after | characters held by those |
|---|---|---|---|---|
| wiki_ar | 436 -> 216 | 28.6% -> 59.7% | 40.3% | 92.3% |
| wiki_ary | 565 -> 499 | 20.6% -> 30.3% | 69.7% | 91.6% |

The footer rule removes 10.1 million characters of wiki_ar (16%) and 0.56 million of wiki_ary
(9%). Four consequences, none yet decided:

- A5 (median length after cleaning >= 300) is **not met by wiki_ar** (216) without a length
  filter.
- A length filter at 300 would keep 40.3% of wiki_ar documents, below A1's 50%, but 92.3% of its
  characters. A1 counts documents, so for Wikipedia it measures the number of stubs rather than
  how much text is lost. A criterion is not changed here: whether to also report retention in
  characters, or to treat stubs differently, is an open decision.
- The rule does not remove empty section headings that sit above the references (the wiki_ary
  day-of-year pages keep "ناس تزادو", "ناس توفاو"); 48 wiki_ar and 2 wiki_ary documents with a long
  line after a footer heading are left uncut on purpose.
- The Google Books stub rule (D-004) drops 382 fw2_apc and 14 fineweb_edu_10bt documents; the 14
  other fineweb_edu pages with one stub sentence have a real book description and are kept.

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

## D-008 (2026-10-28): review of the wiki_footer rule and of the short Wikipedia documents

Follows D-005. Instruction followed: A1 and A5 keep their definition and their limit, and no
300-character filter is added to wiki_ar before the samples were read.

**What wiki_footer removes** (train split; all documents checked by pattern, then read):

- It cut 46,406 of 47,986 wiki_ar documents and 7,405 of 7,654 wiki_ary documents.
- 538 wiki_ar documents (1.2%) and 11 wiki_ary documents lost a line longer than 120 characters.
  Of those I read, most were bibliography entries in other languages (7 of 8 in wiki_ar, 5 of 6
  in wiki_ary); the exceptions were an Arabic dictionary entry and a paragraph about the Tahdar
  reserve. On eight and six documents this only supports a rough bound of about 0.1% to 0.2% of
  documents losing a real paragraph; it is not a precise rate.
- 75 wiki_ar documents (0.2%) have a short heading followed by a long line inside the removed part
  (for example "discussions", "further reading"); the ones read were citations.
- Left in place, not caused by the rule: 12.3% of cut wiki_ar documents and 14.0% of wiki_ary end
  in a short heading with nothing under it ("gallery", "medals", "season results", the wiki_ary
  day-of-year pages), because the tables and images were lost in the dump. 301 wiki_ar documents
  (0.6%, about 0.6% of the kept characters) and 24 wiki_ary documents contain raw wiki table
  markup (`{| class="wikitable"`, `|-`, `|}`).

**Decision:** keep `wiki_footer` as written. Its measured cost (about 0.1% to 0.2% of documents
losing a paragraph) is small next to what it removes. The empty trailing headings and the table
markup are recorded as possible later rules; neither is written, because each is under 15% of
documents and under 30 characters per document (headings) or 0.6% of characters (markup).

**The short documents after the footer is removed** (documents under 300 characters):

| | documents | median | in template clusters* | in clusters of 5 or more |
|---|---|---|---|---|
| wiki_ar | 28,635 (59.7% of train) | 130 chars | 30.2% | 24.4% |
| wiki_ary | 2,317 (30.3%) | 210 chars | 26.0% | 23.7% |

\* MinHash on 3-word shingles, Jaccard >= 0.5, cluster of 2 or more, inside a sample of 50,000
(wiki_ar) and 8,000 (wiki_ary) documents. A full corpus has more members per template, so these
are lower bounds.

What reading 22 of each showed:

- wiki_ar: generated stubs that differ only in the name and numbers (villages of Yemen, Morocco and
  Egypt with a census figure, footballers, politicians, minor planets, species), and one-fact
  stubs that are not templates (a film, a bilateral-relations page, an animal order). The largest
  clusters hold 900, 661 and 470 documents.
- About 8.4% of wiki_ar documents (7.1% of the short ones) show a **missing template value**:
  "born on 15 October 1964 in  in Mexico. He was elected ." The count comes from a regular
  expression, so it is an estimate. The stub is grammatical Arabic with the facts removed.
- wiki_ary: year pages ("1782 is a year in the Gregorian calendar ..."), day-of-year pages,
  village stubs, disambiguation pages ("this is a disambiguation page, the word X can mean"),
  and real one-sentence definitions of people, bands and events.

**What this changes in the three open questions:**

- The 300-character threshold would remove the one-fact stubs and the genuine short definitions
  together with the generated pages, so a length filter does not separate what we want to
  separate. The template clusters do: at least a quarter of the short documents.
- Removing train documents that duplicate an evaluation document is still mandatory (D-005), and
  most of the largest clusters are the same templates.
- Not decided, with the options: (1) no length filter, report A5 for wiki_ar as failing; (2) cap
  the documents kept per template cluster, then look at A5 and A1 again (not yet measured);
  (3) the 300-character filter, which keeps 40.3% of wiki_ar documents and 92.3% of its
  characters.

**A1 and A5 re-measured with the rules written so far** (`scratchllm data clean-report`; only
the Google Books and footer rules exist, so every A1 figure is provisional and can only fall as
more rules arrive):

| | A1 retention | A5 median after | A5 (>= 300) |
|---|---|---|---|
| fw2_arb, arz, ary, ars, acm | 100% | 1564, 1166, 1094, 1363, 991 | pass |
| fw2_apc | 96.0% | 773 | pass |
| fineweb_edu_10bt | 100.0% | 2933 | pass |
| wiki_ar | 100.0% | **216** | **fail** |
| wiki_ary | 99.9% | 499 | pass (evaluation-only, D-002) |

Nothing changed from the previous measurement because the rules did not change; the table is
here as the baseline against which the next rule is judged.

## D-009 (2026-10-28): capping template clusters does not fix A5, and what it does buy

Follows D-008, option 2. Measured on the cleaned (footer removed) train split: documents are
clustered by MinHash on 3-word shingles at Jaccard >= 0.5; each cluster keeps its `cap` longest
documents (ties by text hash, so the choice does not depend on file order).

wiki_ar (47,986 documents; 990 clusters holding 21.1% of documents; largest clusters 1,029, 900,
694, 470; 114 clusters of 10 or more):

| cap per cluster | documents kept (A1) | characters kept | median | under 300 | A5 |
|---|---|---|---|---|---|
| none | 100.0% | 100.0% | 216 | 59.7% | fail |
| 50 | 90.5% | 96.7% | 221 | 57.7% | fail |
| 10 | 87.1% | 95.9% | 226 | 56.5% | fail |
| 5 | 85.5% | 95.6% | 231 | 55.8% | fail |
| 3 | 84.2% | 95.4% | 235 | 55.2% | fail |
| 1 | 81.0% | 94.9% | 251 | 53.7% | fail |

(wiki_ary, evaluation-only: 29.0% of documents in 82 clusters, one of 1,233 documents; with cap 5,
74.2% of documents and 85.2% of characters kept, median 549, A5 passes either way.)

**Correction of the earlier recommendation.** D-008 said capping targets the real cause. It
targets the generated duplicates, but they are not why wiki_ar fails A5: even keeping one document
per cluster the median is 251 and 53.7% of documents are still under 300 characters. Most short
documents are one-fact stubs that are not templates (D-008).

**What the cap does buy:** the clusters hold 14.5% of documents but only 4.4% of characters at
cap 5, because they are short. So it barely changes how much text the model sees. It matters for
evaluation hygiene (the same templates sit in train and evaluation, D-005) and for not teaching
the model one sentence shape a thousand times, and that is the case for it, not A5.

**What this leaves for A5:** the three honest options are (1) keep A5 failing for wiki_ar and say
so in every result; (2) change what A5 measures for Wikipedia, which needs its own dated decision
because it changes a pre-registered criterion; (3) the 300-character filter (40.3% of documents,
92.3% of characters). A5's purpose was to avoid wasting the context window on short documents;
if documents are packed end to end with a separator (the choice question 2 of the EDA plan
leaves open), a short document costs nothing in padding, which is an argument for (1) or (2) and
against (3). That packing choice has not been made.

**Decision recorded now:** none on A5 or on a cap value. The cap is not implemented. Whether to
implement it at all, and at which value, is decided together with the purge of train documents
that duplicate evaluation documents.

## D-010 (2026-10-28): A5 for wiki_ar is settled with the pre-registered remedy, and A1 fails by exception

Follows D-008 and D-009. A5 and A1 keep their wording and their limits.

**Decision:**

- `wiki_ar` gets a minimum length of 300 characters of prose, measured after the footer rule
  (`min_length` in `src/scratchllm/data/cleaning.py`, value in `MIN_PROSE_CHARS`). This is the
  remedy A5 names for a failing median, applied as written.
- It applies to the training use of `wiki_ar`, and to its validation and test splits, so that
  perplexity is measured on the kind of text the model is trained on. `wiki_ary` and the other
  evaluation-only sources (D-002) get no length filter, as D-005 said.
- A1 for `wiki_ar` fails by document count and stays recorded as failing. It is not redefined.
  What A1 prescribes for a failure ("inspect by hand, drop unless a cleaning rule fixes it") was
  done in D-008: the removed documents are generated pages and one-fact stubs, about 8% of them
  with missing template values, not junk text, and the source is not dropped. This is an
  **exception to A1's consequence**, written here, not a change to A1.

**Measured** (`scratchllm data clean-report`, train split of the sample):

| wiki_ar | before the rules | after footer + min_length |
|---|---|---|
| documents | 47,987 | 19,351 (A1 retention **40.3%**, below 50%: fail) |
| median length | 436 | **1,259** (A5 >= 300: pass) |
| under 300 characters | 28.6% | 0.0% |
| characters | 63.3 M | 49.1 M (92.3% of what remained after the footer rule) |

The length rule alone removes 28,635 documents and 4.1 million characters (7.7% of the prose that
was left).

**Why this and not the alternatives of D-009:**

- Keeping A5 failing would have left in the stubs with missing template values for the sake of a
  packing argument; packing removes the padding cost of short documents, not their quality problem.
- Redefining A5 or A1 for Wikipedia would be changing a pre-registered criterion after seeing the
  result for one source.
- The cost is small: 7.7% of the characters of a source that is secondary to fw2_arb.

**What this does not settle:**

- Whether 300 is the right value, and whether the filter helps at all. A data filter can be
  tested, unlike a lossy normalization (D-007): train with and without it and compare held-out loss
  on one fixed evaluation set. Until a model exists the value is a default, not a finding.
- It also removes real one-sentence definitions (a film, a band, an animal order) with the
  generated pages. That is the cost accepted above.
- The cap per template cluster (D-009) and the purge of training documents that duplicate
  evaluation documents (D-005) are separate and still open; the purge is mandatory.
- The packing decision (question 2 of the EDA plan) is still open; it no longer decides A5.

## D-011 (2026-10-28): personal-data masking is implemented, and A7 is corrected for placeholder emails

Follows D-001 and D-003. Report `final-v2` replaces `final-v1` for A7; `final-v1` stays in git as
the earlier measurement. Every other number in the two reports is identical, including the
near-duplicate fingerprints.

**A measurement error found while reading the masked spans.** 89% to 100% of the emails the check
found are `email@example.com` and `firstname.lastname@example.org`: the FineWeb pipelines already
replaced real emails by these. The domains example.com, .org and .net are reserved (RFC 2606) and
cannot be a person's address, so they are not personal data and were wrongly counted. They are now
reported apart (`email_placeholder_matches`) and are in no A7 figure. This is a correction of what
was counted, not a relaxation of the limit, but it changes a measured value of a pre-registered
criterion, so it is recorded here.

A7, documents with a real email or phone-like number per 1,000 (limit 5):

| | D-003 (final-v1) | corrected (final-v2) | A7 |
|---|---|---|---|
| fw2_arb | 23.3 | 13.7 | fail |
| fw2_arz | 20.4 | 17.8 | fail |
| fw2_ary | 5.6 | **3.2** | **pass** (was fail) |
| fw2_ars | 51.2 | 40.8 | fail |
| fw2_apc | 8.0 | 6.0 | fail |
| fw2_acm | 30.7 | 25.3 | fail |
| fineweb_edu_10bt | 41.6 | 26.4 | fail |
| wiki_ar, wiki_ary | 0.3, 0.3 | 0.3, 0.3 | pass |

What is left is almost all phone numbers. Real emails are 0 to 8 per source (3 in fw2_arz, 6 in
fw2_arb), 145 in fineweb_edu_10bt, whose pipeline anonymized differently.

**Decision (D-003 stands, as code):** `pii` in `src/scratchllm/data/cleaning.py`, with detection in
`src/scratchllm/data/pii.py` shared with the A7 check. Each email or phone-like number becomes
`<email>` or `<phone>`; a page with five or more real matches is dropped; placeholder emails are
masked too (one token for "an email") but do not count towards the five. Web sources only. It is
applied to fw2_ary as well although that source now passes A7, so that all web sources are cleaned
the same way (53 of its documents change, 1 is dropped).

**Measured effect** (`scratchllm data clean-report`, train split; retention is documents kept):

| | retention | documents masked | dropped as spam | characters removed |
|---|---|---|---|---|
| fw2_arb | 99.9% | 1,088 | 37 | 464,842 |
| fw2_arz | 99.8% | 895 | 83 | 533,215 |
| fw2_ary | 100.0% | 53 | 1 | 2,490 |
| fw2_ars | 99.8% | 469 | 19 | 204,755 |
| fw2_apc | 96.0% (382 are Google Books stubs, D-004) | 77 | 0 | 900 |
| fw2_acm | 99.8% | 188 | 10 | 71,448 |
| fineweb_edu_10bt | 99.9% | 1,947 | 45 | 1,004,818 |

The median length moves by 4 characters at most. A7 after masking is 0 in every source, which is
true by construction and proves nothing about what the patterns miss.

**What reading showed** (the check D-003 named as the real test):

- Precision: 80 random masked spans, 20 from each of fw2_arz, fw2_ars, fw2_arb and
  fineweb_edu_10bt. About 5 were not phone numbers: two ISBN-10 numbers after "ISBN" (fineweb_edu),
  a photo id inside a URL, "03 2011 450" from a date and time, and a row of zeros. One more was a
  bank account number. Many of the correct hits are business numbers (hotlines, shops), which are
  masked too. Five of 80 is roughly 6%, from a small sample.
- Recall is not estimated. Of 30 digit runs of 8 or more digits that were not matched, most were
  dates, post timestamps, year ranges and Facebook or image ids, which should stay, and 2 looked
  like phone numbers: a landline written without its leading zero ("112658888") and an
  eight-plus-eight grouping ("22601858- 22601859"). The sample cannot give a rate.

**Decision on the known errors:** none are patched now. Changing the patterns changes the A7
measurement again and means another report; candidates, in the order of their likely gain, are
excluding numbers after "ISBN", excluding digit runs inside URLs, and a rule for landlines without
a leading zero when a word such as "phone" or "call" precedes them. They are listed, not written.
