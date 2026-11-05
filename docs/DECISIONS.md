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

## D-012 (2026-10-28): train/evaluation leakage measured across all sources, and what the samples cannot show

Follows D-005 (A8, A9). Nothing was removed or rewritten; the data files were hashed before and
after the run and are identical. Tool: `scratchllm data leakage-report`
(`src/scratchllm/eda/leakage.py`). The report is tracked as `reports/eda/final-leakage-v1` (no raw text in it).

**Review of the earlier measurement.** The 38 of 2,013 wiki_ar evaluation documents (1.9%) in
D-005 came from `NearDuplicates`, which (a) accepts a pair on the 128-number signature estimate
and (b) joins pairs into clusters, so a chain counts documents that are not themselves similar.
That fits A2, which asks how much of a sample is near-duplicate, but not the question here: does
this evaluation document have a neighbour in that pool. The new tool keeps the training pool and
the queries apart, accepts a candidate only on the exact Jaccard similarity, and compares a query
with up to 50 pool documents per LSH bucket. On the same raw wiki_ar text it finds **21 of 2,013
(1.04%)**, not 38. The D-005 figure overstated the rate; the A9 conclusion (above the 0.5% limit
on raw text) stands.

**Validation of the algorithm before using it:**

- Signatures are identical to those of `NearDuplicates` (unit test).
- All pairs of wiki_ary (7,993 documents): 15 pairs have exact Jaccard >= 0.8 and LSH offered all
  15 as candidates. The signature estimate is off by 0.029 on average (95th percentile 0.072). Of
  the pairs it called >= 0.8, only 33% (15 of 45) really are; this is why the estimate alone is
  not used to decide.
- Planted copies of 1,500 real fw2_arz documents with random word replacements, found when LSH
  offers the true source and the signature agreement is at least 0.5 (grouped by true Jaccard):
  [0.9, 1.0] 335 of 335; [0.8, 0.9) 233 of 238 (97.9%); [0.7, 0.8) 153 of 182 (84.1%);
  [0.6, 0.7) 55 of 144 (38.2%); [0.5, 0.6) 14 of 117; below 0.5 almost none. Right at 0.8 the
  recall is about 95% (the banding's theoretical value is 94.8%), so a count at the threshold is
  a slight undercount. This validation uses the same signatures and bands as the tool, not the
  tool itself.

**Result** (report `reports/eda/final-leakage-v1`, all nine samples, text after the cleaning rules; every count below is taken from `leakage.json`, which also checks each source against its manifest: `all_counts_reconcile` is true). The training pool is the train split of fw2_arb (48,232), fw2_arz (47,900), fw2_ary (9,649), fw2_ars (9,507), wiki_ar (19,351), fineweb_edu_10bt (47,880), 182,519 documents. Evaluation is the val and test of those sources, 7,449 documents, and every document of fw2_acm, fw2_apc and wiki_ary, 24,601 documents.

| evaluation group | documents | exact duplicates (A8) | Jaccard >= 0.8 (A9) | >= 0.6 |
|---|---|---|---|---|
| fw2_arb (val+test) | 1,730 | 0 | 0 | 0 |
| fw2_arz (val+test) | 2,014 | 0 | 0 | 1 |
| fw2_ary (val+test) | 350 | 0 | 0 | 0 |
| fw2_ars (val+test) | 473 | 0 | 0 | 0 |
| wiki_ar (val+test) | 823 | 0 | 0 | 3 |
| fineweb_edu_10bt (val+test) | 2,059 | 0 | 0 | 0 |
| fw2_acm (evaluation-only source) | 6,990 | 0 | 0 | 0 |
| fw2_apc (evaluation-only source) | 9,618 | 0 | 0 | 0 |
| wiki_ary (evaluation-only source) | 7,993 | 0 | 0 | 2 |

Same measurement on wiki_ar with the cleaning rules switched on one at a time (evaluation
documents with a neighbour at Jaccard >= 0.8 / >= 0.6). The first two rows come from a separate
run of the same tool, not from the report, which always applies all the rules; the last row is the
report's own wiki_ar line:

| wiki_ar text | evaluation documents | >= 0.8 | >= 0.6 |
|---|---|---|---|
| raw | 2,013 | 21 (1.04%) | 144 (7.15%) |
| category footer removed | 2,013 | 4 (0.20%) | 100 (4.97%) |
| footer removed and 300-character minimum (D-010) | 823 | 0 | 3 (0.36%) |

Most of the raw-text similarity was the shared category list, not the prose. The footer rule alone
brings wiki_ar under the A9 limit of 0.5%, which is a reason for it independent of D-005's.
The 0.6 to 0.8 band (5% of wiki_ar evaluation documents after the footer rule, 0.4% after the
length filter) is the same generated templates with different names; A9 does not count it, but a
perplexity measured on such documents partly measures a memorized template.

**What the samples cannot show.** Each sample is a small share of its source:

| source | rows in the source | sampled | share |
|---|---|---|---|
| fw2_arb | 61,977,525 | 50,000 | 0.08% |
| fw2_ary | 2,365,405 | 10,000 | 0.42% |
| fineweb_edu_10bt | 9,672,101 | 50,000 | 0.52% |
| fw2_ars | 298,167 | 10,000 | 3.35% |
| wiki_ar | 1,219,201 | 50,000 | 4.10% |
| fw2_arz | 853,290 | 50,000 | 5.86% |
| fw2_apc | 69,740 | 10,000 | 14.34% |
| fw2_acm | 7,382 | 7,000 | 94.83% |
| wiki_ary | 8,087 | 8,000 | 98.92% |

An evaluation document whose twin sits in the full training corpus has about the share above as
its chance that the twin is also in our sample (0.08% for fw2_arb). So the zero found between
sources says the tool works and that Wikipedia templates no longer leak after the rules; it does
not say the corpus is free of cross-source copies. Within one source, templates have many members
and are visible; one-off copies in a large crawl are not. A8 and A9 are therefore passed on the
samples and **not established for the corpus**.

**Decision:**

- The purge of training documents that duplicate evaluation documents (D-005) stays mandatory and
  is a step of the corpus build, run against the training documents actually selected, not
  against these samples. Nothing is removed now.
- Its design follows from the table above: the evaluation side is small and almost complete
  (fw2_acm 94.8%, wiki_ary 98.9%), the training side is huge. So the LSH index is built over the
  evaluation documents only and the training corpus is streamed once through it, each training
  document checked by exact Jaccard against its candidates. Memory is that of the evaluation set;
  time is one signature per training document (about a millisecond), i.e. an hour for a few
  million documents, which is the order of the training set.
- The tool of this entry is the reference for that step. Known limits to carry over: the
  50-document bucket cap, recall of about 95% right at 0.8, word-shingle similarity only (a
  rewritten or translated copy is not found).
- A9 on a template family at 0.6 to 0.8 is not a criterion; whether to also report it stays open.

## D-013 (2026-10-31): normalization = presentation forms, tatweel, whitespace (`tokenizer/normalize.py`)

Implements the "applied to every source" part of D-006; nothing lossy was added.

- **NFKC only on Arabic presentation forms** (U+FB50-FDFF, U+FE70-FEFF), not on the whole text.
  Measured on the nine samples: those characters are 0.04% of fw2_arb but appear in 6% of
  fw2_apc documents (616 of 9,670). NFKC on everything would also have changed over 59,000 "…",
  1,500 "²" and 1,200 "™" in the English text for no gain.
- **Order: NFKC, then tatweel, then whitespace.** NFKC of ten presentation forms contains a
  tatweel (U+FE77 becomes tatweel + fatha), so removing the tatweel first leaves stray ones and
  breaks idempotence (a deliberately reordered version fails two tests).
- **Whitespace:** runs of spaces, tabs and no-break spaces become one space, spaces around a
  newline go, three or more newlines become two, the ends are stripped. Line and paragraph
  structure stay.
- **Untouched:** digits (NFKC leaves Arabic-Indic digits as they are, so the digit decision of
  D-006 stays open), diacritics, alef/yeh/teh-marbuta spellings.
- **Checked on real text:** on 117,000 documents without presentation forms, the text differs from
  the original only in whitespace and tatweel; no document became empty; applying it twice gives
  the same result. It removes 0.00% to 0.73% of the characters of a source and changes 99% of the
  Wikipedia documents (double spaces and spaces before newlines).
- **Not decided (counted but not acted on):** zero-width and direction marks, and the BOM.

## D-014 (2026-11-01): pre-tokenizer = GPT-2's pattern plus combining marks; byte-level BPE trainer

`tokenizer/pretokenize.py` and `tokenizer/bpe.py`. The trainer does not encode or decode yet.

- **Base units are bytes (256 ids), as in GPT-2.** Any text can be written; there is no unknown
  token. The cost for Arabic is that a letter is two bytes, so the first merges rebuild letters.
  Measured in a probe run (4,096 ids, 2M characters from each of six training sources after
  cleaning and normalization): 11 of the 3,837 learned tokens (0.3%) are not valid UTF-8, so
  this waste is small. Character-level BPE was not tried; this is a measurement of the byte
  version, not a comparison.
- **The pre-tokenizer is GPT-2's pattern with `\p{M}` added to what a word is.** GPT-2's `\p{L}+`
  cuts a diacritized Arabic word at every diacritic. On 1,500 documents per source this adds
  9.2% more pieces in fw2_arb, 3.8% in wiki_ar, 2.8% in fw2_arz and 0.02% in fineweb_edu.
- **Kept as in GPT-2 on purpose:** a paragraph break ("\n\n") before a word is two pieces, because
  only a space can be glued to a word. Treating a newline run as one piece would save 1.9% of
  wiki_ar's pieces and nothing in fw2_arb or fineweb_edu (they have almost no "\n\n"). Digits are
  runs of any length (GPT-2); splitting them one by one is not decided. The English contractions
  are in the pattern; the curly apostrophe is not.
- **Ties** between pairs with equal counts go to the smaller ids. Arbitrary, but the same corpus
  always gives the same merges.
- **Special tokens are cut out of the text before counting** (`<|endoftext|>`, `<email>`,
  `<phone>`), so they never take part in a merge; their ids will be the last ones of the vocabulary.
- **Correctness:** the incremental trainer (pair counts updated only for words that contained the
  merged pair, a heap for the best pair) gives the same merges as a slow reference that recounts
  everything, on 8 random corpora with a tiny alphabet (many ties) and on Arabic text. Three
  deliberate breakages (new pairs not indexed, counts not pushed to the heap, ties to the larger
  pair) each fail tests.
- **Cost measured:** 275,512 unique pre-tokens from 12M characters; 3,837 merges in 19 seconds;
  peak memory 474 MB. Memory grows with the number of unique pre-tokens, and the machine has
  about 1 GB free, so the training sample for the 16k vocabulary must be chosen with this in mind.
- **Not decided:** the vocabulary size (16k is a proposal, to be chosen by fertility and
  bits-per-char), how many characters per source go into tokenizer training, and whether the
  tokenizer sample should be weighted by source rather than equal per source.

## D-015 (2026-11-01): encoding replays the merges in learned order; the tokenizer file (`tokenizer/tokenizer.py`)

- **Id layout:** 0-255 the bytes, then one id per merge in the order learned, then the special
  tokens in the order given. The vocabulary size is 256 + merges + specials.
- **Encoding rule:** within a pre-token, apply the merge that was learned earliest among the pairs
  present, until none is left. Not "the most frequent pair in this word": ranks come from the whole
  corpus. Checked against the literal definition (apply merge 0, then merge 1, ...) on random text
  with a trained tokenizer, and by a case where the leftmost pair loses to an earlier-learned one.
- **Special tokens in text:** by default the strings `<email>`, `<phone>`, `<|endoftext|>` found in
  the text become their reserved ids, because cleaned training text contains `<email>` and `<phone>`
  (D-011). `parse_special=False` treats them as ordinary characters, for untrusted user input.
- **No normalization inside the tokenizer.** The text given to `encode` is encoded as it is
  (D-013). The data pipeline and any prompt must call `normalize` first; the tokenizer does not
  enforce it.
- **Decoding** returns the exact text for any ids made by `encode`. Ids a model generates can end
  in the middle of a character (an Arabic letter is two bytes), so `decode` replaces such a
  fragment with U+FFFD by default and can be made strict. An id outside the vocabulary raises.
- **File:** JSON with format/version, the pre-tokenizer pattern, the merges and the special
  tokens; same bytes every save; written to a temporary file and renamed. Loading refuses a file
  whose pattern differs from the code's, because the same merges with another pre-tokenizer would
  not reproduce the training.
- **Checked on real text** with a 4,096-id probe tokenizer (2M characters per source for
  training): 300 validation/test documents of each of the six training sources, and 300 documents
  of each evaluation-only source it never saw (fw2_apc, fw2_acm, wiki_ary): 2,700 documents, cleaned and
  normalized and also raw, 0 round-trip failures. Speed 1.0 to 3.0 million characters per second
  in one Python process. Characters per token at this size: 2.2 to 2.9 across sources. That is a
  probe of the code, not the fertility result of step 3; the vocabulary size is still open (D-014).

## D-016 (2026-11-01): tokenizer fertility measured against GPT-2 and Qwen3; sample size and vocabulary options

Code: `tokenizer/corpus.py`, `fertility.py`, `report.py`, commands `scratchllm tokenizer train` and
`scratchllm tokenizer fertility`. The report is `reports/tokenizer/final-v1` (tables generated from
`report.json`; sha256 of the tokenizer file `698bb54137491734`). Tokenizer file:
`artifacts/tokenizer/bpe-32k.json` (not tracked in git yet).

- **How it was measured.** Up to 600 held-out documents per source (val+test
  of the six training sources; all documents of the three evaluation-only sources, which the
  tokenizer never saw), cleaned and normalized, the same text for every tokenizer. Tokens per word
  uses `str.split()` as the word; characters per token does not depend on that. GPT-2 and Qwen3
  (`Qwen3-4B-Instruct-2507`) are measured on the same text with the `tokenizers` library.
- **Training sample.** Equal budget per training source (fw2_arb, fw2_arz, fw2_ars, fw2_ary,
  wiki_ar, fineweb_edu): 3,000,000 characters of train text each, 366,966 distinct pre-tokens, one
  training to 32,768 ids, from which every smaller size is a prefix of the merges
  (`Tokenizer.truncated`; the prefix property is a test). So English is 1/6 of the training
  characters, and Arabic 5/6. No other weighting was tried.

| tokenizer | vocab | Arabic tokens/word | Arabic chars/token | English tokens/word | English chars/token |
|---|---:|---:|---:|---:|---:|
| Qwen/Qwen3-4B-Instruct-2507 | 151669 | 2.39 | 2.35 | 1.31 | 4.55 |
| openai-community/gpt2 | 50257 | 5.51 | 1.02 | 1.32 | 4.50 |
| ours-16384 | 16384 | 1.78 | 3.17 | 1.78 | 3.33 |
| ours-24576 | 24576 | 1.68 | 3.36 | 1.67 | 3.57 |
| ours-32768 | 32768 | 1.62 | 3.48 | 1.60 | 3.72 |
| ours-8192 | 8192 | 1.97 | 2.86 | 2.03 | 2.92 |

- **Reading it.** On Arabic, GPT-2 needs about 3.1 times as many tokens as ours at 16,384 ids
  (5.51 vs 1.78 per word, and 1.02 characters per token: essentially one token per letter), and
  Qwen3, with a vocabulary 9 times larger, about 1.34 times as many (2.39 vs 1.78). On English the
  order reverses: ours needs 1.78 tokens per word against 1.32 for GPT-2 and 1.31 for Qwen3.
  The dialects the tokenizer never saw (fw2_acm, fw2_apc, wiki_ary) are 1.78 to 2.01 at 16,384
  ids, somewhat higher than the trained ones (1.63 to 1.82) but still far below GPT-2 and Qwen3,
  so most of what was learned carries over to dialects outside the training sources.
- **Sample size.** A tokenizer with 16,384 ids trained on 1,000,000 characters per source instead of
  3,000,000 differs by at most 0.03 tokens per word on any of the nine sources (scratch run, not
  in the repo), so the sample size is not what limits quality. Peak memory was about 0.5 GB with
  1M per source and 0.9 GB with 3M.
- **Vocabulary size is NOT decided here.** Doubling 8,192 to 16,384 ids lowers the Arabic mean by
  9.7%; doubling 16,384 to 32,768 by 8.7%: the gain does not flatten yet. The cost is the embedding
  table: 6.3M parameters at 16,384 ids and d_model 384 (12.6M at 32,768), against a model of 10 to
  30M parameters in total. Which size trains better per parameter and per token can only be
  decided by training with each (bits per character, D-007), so 16,384 stays the proposal.
- **Limits.** English is clearly worse than the outside tokenizers; whether to give English a
  larger share of the tokenizer sample is untested. The special tokens `<email>` and `<phone>`
  are 0.0028% of the evaluation tokens, so they do not explain any difference. "Word" is a rough
  unit for Arabic. The held-out sample is the first 600 documents of each
  file, not a random draw.

## D-017 (2026-11-02): Wikipedia pages are excluded from the web sources; the 16,384-id tokenizer; token files and the window dataset

Decided by the user: vocabulary 16,384; exclude `*.wikipedia.org` from the web sources only (the
Wikipedia sources keep their pages); keep the 32k tokenizer and its provenance in git.

- **Exclusion (`cleaning.py`, rule `wikipedia_page`, a new kind of rule that looks at the record's
  url, `DocRule`).** Applied to every web source, evaluation-only ones included. Dropped in the
  samples: fw2_arb 102 documents, fw2_arz 308, fw2_ary 22, fw2_apc 1, fineweb_edu 446 (441 of them
  in train), fw2_ars 0, fw2_acm 0; retention stays at 99.0% to 99.8% for these sources except
  fw2_apc (96.0%, from the books-stub rule of D-004). All but five of the 879 pages are on the
  train side, because the split key of a web document is its domain.
- **Why, and what is NOT shown.** It is a precaution against the same article being a training
  document of a web source and an evaluation document of wiki_ar. It is not a measured problem:
  none of the 122 `ar.wikipedia.org` pages in the web samples has the same title as a document
  of the wiki_ar sample (which covers 4.1% of wiki_ar, so this cannot rule it out in the full
  corpus), and the near-duplicate measurement of D-012 found none either. A copy of an article on
  another site (a mirror) is not found by url.
- **Tokenizer retrained with the exclusion.** The first 39 of 32,509 merges are the same, the rest
  differ (different training text, so different counts and ties). Fertility changes by at most
  0.003 tokens per word and 0.006 characters per token on the language means, for every
  size of ours
  (`reports/tokenizer/final-v2`, against `final-v1`; the evaluation text lost only the five
  Wikipedia documents of fineweb_edu's val/test). At 16,384 ids: Arabic 1.78
  tokens per word, English 1.78; the rest of D-016 holds.
- **Files in git.** `artifacts/tokenizer/bpe-32k.json` and `.provenance.json` (final-v2; the
  version measured in final-v1 is in the history, commit 0c56002), and `bpe-16k.json` with its
  provenance. The 16k file has exactly the first 16,125 merges of the 32k file (checked), so the
  32k file can produce any size up to 32,768 with `Tokenizer.truncated`.
- **Token files (`data/tokens.py`, `scratchllm data tokenize`).** One uint16 file per source and
  split in `data/processed/tokens-16k/` (not in git: 364 MB), the `<|endoftext|>` id after every
  document, a `<source>.tokens.json` per source with counts, sha256 of each file, the tokenizer's
  sha256, the cleaning rules and the sha256 of the sampling manifest. Training sources get train,
  val and test; the three evaluation-only sources get one `eval` file. Totals: train
  167,394,059 tokens in 181,648 documents; val 3,344,249
  (3,620); test 3,647,361 (3,824); eval-only 12,175,680
  (24,600). `scratchllm data verify-tokens` re-hashes the files.
- **Checked.** Every id is below the vocabulary size; the number of end ids equals the number of
  documents in every file; the three files of fw2_ary are identical, id for id, to an independent
  re-encoding of the cleaned text with the plain tokenizer; 19.5 minutes and 152 MB of memory for
  all nine sources. Six deliberate breakages of the writer and the dataset (target not shifted,
  window count off by one, windows leaking into the next file, no end id, eval-only sources split,
  wrong dtype) each fail tests.
- **Dataset (`training/dataset.py`).** Book-style sliding windows: input = ids[s:s+T], target =
  ids[s+1:s+T+1]; `stride` defaults to T (no overlap); a window never crosses from one file to the
  next; files are read with `numpy.memmap`; the loader's order depends only on its seed.
- **Not decided.** How sources are mixed during training (concatenating the files gives each
  source a share proportional to its tokens: fineweb_edu is 39% of train tokens, fw2_arb 23%,
  fw2_arz 18%, wiki_ar 9%, fw2_ars 8%, fw2_ary 3%); the context length and the stride; whether to
  purge near-duplicates (measured only, D-012). **A fact for planning:** the train split is
  167M tokens; a model of 20M parameters sees about 8.4
  tokens per parameter in one pass, against the rule of thumb of about 20 (Hoffmann et al. 2022,
  from memory, to be checked), so a run on this sample means several passes or more data.

## D-018 (2026-11-02): causal multi-head self-attention (`model/attention.py`)

Chapter 3 of the book. The explanation of the mechanism (query, key, value, scaling, mask, heads)
is in the module's docstring; this entry records the choices.

- **One fused matrix for the three maps** (`d_model x 3*d_model`), as in nanoGPT, instead of three
  matrices as in the book. The same computation in one matrix product; a test checks that its
  three thirds give q, k and v.
- **Explicit math is the reference.** `softmax(q k^T / sqrt(head_dim))` with the future set to
  -inf, then `weights v`, heads put side by side, one output map. It can return the attention
  weights. It matches both the library kernel (`F.scaled_dot_product_attention`) and a plain
  triple loop over examples, heads and positions, and changing the future never changes the past
  (tested at three cut points).
- **`fast=True` uses the library kernel** with the same parameters; the tests check the same
  numbers, causality, gradients and that dropout acts only in training. Measured on the RTX 3050
  for one attention layer (d_model 384, 6 heads, 512 tokens, forward and backward, bfloat16):
  batch 16: 26.7 ms and 442 MiB explicit against 6.6 ms and 125 MiB with the kernel; batch 32:
  45.3 ms and 862 MiB against 12.7 ms and 227 MiB. The explicit version builds a
  batch x heads x T x T table, so stacking layers would not fit in 4 GiB at larger batches; the
  model will use `fast=True` for training. These are one layer's numbers, not a model's.
- **No bias by default** (GPT-2 has biases): fewer parameters, and the modern default. It is an
  option, not a measured choice; `4 * d_model^2` parameters per layer without bias.
- **The causal mask is a non-persistent buffer** (rebuilt from `context_length`, not saved in
  checkpoints). A sequence longer than `context_length` is refused.
- **Positions are not in this module.** Attention alone does not know token order. Learned
  absolute position embeddings (GPT-2, the book) against rotary embeddings is the next
  architecture decision, to be taken when the model is assembled (chapter 4).
- Mutation checks: eight deliberate breakages (no mask, mask also hiding the diagonal, scaling by
  `d_model`, softmax over the wrong axis, heads merged without transpose, values taken from the
  keys, fast path not causal, fast path dropping out in eval) each fail tests.

## D-019 (2026-11-02): the GPT model (`model/layers.py`, `model/gpt.py`)

Chapter 4 of the book. The mechanism of each part (layer norm, GELU, feed-forward, pre-norm
residual blocks, position embeddings, weight tying, initialization) is explained in the module
docstrings; this entry records the choices and what was measured.

- **Architecture = GPT-2's:** token + learned absolute position embeddings, `n_layers` pre-norm
  blocks (`x + attention(norm(x))`, `x + feed_forward(norm(x))`), a final layer norm, an output
  map tied to the token embedding. Rotary embeddings were not used: learned positions are what the
  book teaches and cost 0.2M parameters at 512 positions. Changing this later is a model change,
  to be tested on bits per character (D-007), not decided here.
- **Layer norm, GELU (tanh form) and feed-forward are written out** and compared with the library
  versions in tests. Layer norm computes in float32 whatever the input dtype, because a variance in
  bfloat16 is too coarse.
- **No bias anywhere by default** (attention, feed-forward, layer norm), following D-018; `bias=True`
  is an option. Feed-forward width is 4 x `d_model`.
- **Initialization** as GPT-2: weights Normal(0, 0.02), the two maps that write into the residual
  stream scaled by 1/sqrt(2 n_layers). Checked: an untrained model's loss is within 0.1 of
  ln(vocab) = ln(1000) on random ids, i.e. it starts as a uniform guess.
- **Parameters.** The default local model (16,384 ids, 512 positions, d_model 384, 6 layers,
  6 heads) has 17.1M parameters, of which the tied embedding is 6.3M (37%); a model with d_model 512,
  8 layers, 8 heads has 33.8M. The count is tested against the formula
  `V d + T d + L (4 d^2 + 2 d d_ff + 2 d) + d`.
- **A claim of mine that was wrong, found by its test.** I expected that without position
  embeddings a model cannot tell a text from its reordering. That is exact for one layer (the last
  token sees only the set of earlier token embeddings), but with two or more layers the causal
  mask alone lets earlier tokens carry order, because each one sees a different prefix; the test
  failed at two layers by about 0.002 in the logits. The docstring and the test now say so (the
  literature reports models without position encodings still learning position; from memory,
  not checked). Position embeddings are the explicit signal, not the only one.
- **Correctness evidence.** Causality of the whole model (changing later ids never changes earlier
  logits, three cut points); `fast_attention` gives the same logits as the explicit path; a
  checkpoint restores identical predictions; dropout acts only in training; a small model
  memorizes a batch (loss from above 3.0 to below 0.3 in 200 steps). Eight deliberate breakages
  (no residual, no position embedding, unscaled residual writers, untied weights, no final norm,
  unbiased variance, wrong GELU constant, post-norm order) each fail tests; the first run of the
  GELU breakage did not apply (it hit the docstring) and was redone on the code line.
- **Training throughput on the RTX 3050 (4 GiB), measured**, bfloat16 autocast, fast attention,
  fused AdamW, 512 tokens per sequence, 5 timed steps after one warm-up:

  | model | batch | ms/step | tokens/s | peak memory |
  |---|---:|---:|---:|---:|
  | 17.1M | 4 | 82 | 25,100 | 1,101 MiB |
  | 17.1M | 8 | 164 | 24,900 | 1,946 MiB |
  | 17.1M | 16 | out of memory | | |
  | 33.8M | 4 | 153 | 13,400 | 1,684 MiB |
  | 33.8M | 8 | 285 | 14,400 | 2,882 MiB |
  | 33.8M | 16 | out of memory | | |

  At 25,000 tokens per second one pass over the 167M training tokens of D-017 takes about 1.9
  hours for the 17.1M model and, at 14,000, about 3.3 hours for the 33.8M one (arithmetic from the
  measured rates, not a measured epoch). The memory limit at batch 16 is the logits: batch x 512 x
  16,384 ids in float32 is 537 MB for one copy, so larger effective batches must come from
  gradient accumulation (or a chunked loss), not from a larger batch.
- **Not decided:** context length (512 is the default, not tuned), the optimizer schedule, batch
  size in tokens, dropout (0 by default), and whether to use `torch.compile`.

## D-020 (2026-11-02): the training pipeline (`training/`, `scratchllm train`)

Chapter 5 of the book, plus what a run needs to be trusted and to survive a stop. Generation and
fine-tuning are not part of it. Modules: `config.py` (typed YAML, unknown keys refused),
`schedule.py`, `batches.py` (sampling and validation), `checkpoint.py`, `logger.py`, `trainer.py`.

- **Optimizer and schedule.** AdamW (betas 0.9/0.95), weight decay only on matrices and
  embeddings, not on norm scales or biases; linear warmup then cosine decay to `min_lr`; global
  gradient-norm clipping (default 1.0, 0 turns it off). The values in `configs/` are common starting
  points, not tuned.
- **Gradient accumulation** divides each micro-batch loss by the number of micro-batches. Adam and
  clipping make the weights insensitive to a wrong scale, so the test also compares the raw
  gradient norm of 2 x 4 sequences with that of 1 x 8 on identical data.
- **Data order is a function of (seed, step, micro-batch)**, not state. Each sequence picks a
  source by weight and a window start uniformly (stride 1) in its train tokens. This makes
  resuming exact without saving a data position. Sampling is with replacement, so there are no
  "epochs"; one pass over D-017's 167M tokens is `max_steps x tokens_per_step` = that number.
- **Source mix.** `data.train_sources` gives explicit weights; omitted, the sources are weighted by
  their token counts (the natural mix, still the open question of D-017).
- **Validation** uses fixed, evenly spaced, non-overlapping windows of each source's val split
  (`eval_batches x micro_batch_size` windows per source), reported per source as loss and bits
  per character, and as an unweighted mean over sources. Bits per character uses the val split's
  tokens-per-character ratio, which includes one end-of-text id per document: an approximation of the
  windows' own density. `best.pt` is written when the mean val loss improves; `last.pt` every
  `checkpoint_interval` steps and at the end, atomically (a failed save keeps the previous file).
- **Resuming** restores weights, optimizer state, step, counters and the random generators. It
  refuses a different model configuration, different token files (by sha256 from their
  descriptions), or a different seed, batch sizes or source weights. Changes to the optimizer,
  schedule or intervals are allowed and logged (`config_sections_changed`), so a run can be
  extended. An existing run directory is never overwritten.
- **Reproducibility, as measured.** On CPU, a run interrupted at step 6 of 12 and resumed gives
  bit-identical weights and identical per-step losses to the uninterrupted run, with and without
  dropout; a crash with no chance to save resumes from the last checkpoint with the same result. On the
  RTX 3050 (smoke config, interrupted at step 15 of 30) the resumed run was also bit-identical in
  this one trial; PyTorch does not guarantee that on GPU, and `deterministic: true` is the setting
  that asks for it (slower; not exercised here).
- **Bad steps.** A step whose loss or gradient norm is not finite changes nothing, is logged as
  `skip`, and still counts in the schedule; `max_consecutive_skips` in a row saves `last.pt`, logs
  `diverged` and raises `DivergedError`. Ctrl-C saves `last.pt` at the last finished step.
- **Log.** `runs/<name>/metrics.jsonl`, one JSON object per line: `start`/`resume` (full config and
  its hash, parameter count, git commit and whether the tree was dirty, library versions, device,
  data hashes), `train` (mean loss since the last line, lr, raw gradient norm, tokens seen, tokens
  per second of compute only, peak GPU memory), `eval`, `checkpoint`, `skip`, `diverged`,
  `interrupted`, `end`. Non-finite numbers are written as strings.
- **Measured on the real files.** `configs/smoke.yaml` (2 layers, 128 wide, 30 steps, 61,440 tokens)
  takes 14 s on the RTX 3050: train loss 9.66 -> 8.62 (a uniform guess is ln 16384 = 9.70),
  val loss 9.15 -> 8.64. `configs/local-17m.yaml` run for 6 steps: 25,000 tokens/s and 2,012 MiB peak
  with micro-batch 8 x accumulation 8 (32,768 tokens per step), matching D-019. No run beyond 6
  steps of the full model has been done, so nothing is known yet about its loss curve.
- **Mutation checks.** Twelve deliberate breakages of the loop and sampler (no loss scaling, no
  zero_grad, learning rate not applied, sampler ignoring the step, generators or optimizer state not
  restored, non-finite steps applied, clipping off, best checkpoint saved on every evaluation,
  data fingerprint unchecked, decay on all parameters, skipped steps not counted) each fail a test;
  two of them first survived (the accumulation scale and the best checkpoint) and the tests were
  strengthened until they failed.
- **Limits.** Single GPU, single process, no `torch.compile`, no learning-rate or batch-size tuning,
  no activation checkpointing, no early stopping. Checkpoints hold the full optimizer state
  (30 MB for the smoke model; about 205 MB for 17M parameters). `deterministic` is untested on GPU.

## D-021 (2026-11-03): pilot run of the 17.1M model, and the ars train/validation gap

`configs/pilot-17m.yaml`: 1,500 steps of 32,768 tokens planned (49M tokens), natural source mix,
the untuned hyperparameters of `local-17m.yaml`. It was stopped on request at step 1,056 (SIGTERM,
which saved `last.pt` at that step; `best.pt` is step 1,000). The schedule did not finish, so
these are mid-schedule numbers.

- **Health.** No skipped steps, gradient norm 0.5 to 0.6, peak memory 2,012 MiB, about 34,500
  tokens/s (D-019 measured 25,000 with shorter timing windows; the difference is not explained).
- **Validation at step 1,000** (mean over the six sources 5.60 nats per token, 2.40 bits per
  character; a uniform guess is 9.70): edu 4.61, wiki_ar 5.27, arb 5.65, arz 5.93, ary 5.86,
  ars 6.30. It was still falling by about 0.07 per 100 steps. Nothing here says anything about the
  quality of the finished model.
- **Train and validation agree within 0.04 on five sources at every evaluation. ars is 0.3 higher
  on validation** (6.30 against a train probe of 6.00 at step 1,000), from step 200 on, not growing.
- **Investigation of that gap** (on `best.pt`, step 1,000; one-off scripts, not kept in the repo):
  - It is not noise in the window sample: all 370 ars val windows give 6.288, train probe 5.950, gap
    +0.337 (standard error 0.025). The control arb gives -0.005. It is not noise in the choice of
    sites either: a bootstrap over domains gives 95% intervals [6.12, 6.37] for ars validation and
    [5.89, 6.02] for its train probe, which do not overlap (arb: [5.63, 5.71] and [5.64, 5.71]).
  - It is uniform over positions inside a window (first 32 tokens +0.31, the rest +0.34), so it is
    not about short context.
  - ars validation is 282 documents from only 39 sites, and one site is 31% of its tokens (three
    sites 58%); its train side is 9,507 documents from 1,886 sites, none shared with validation
    (the split is by site). arb validation has 349 sites. The six largest ars validation sites have
    mean loss 6.05 to 6.59, the six largest train sites 5.56 to 5.93.
  - Memorization is small. Replaying the sampler (it is a function of seed and step) gives the
    5,003 ars windows trained on (17% of its train tokens, 1.1 exposures each). Windows from the
    same train sites that overlap none of them score 5.992, against 5.892 for the windows trained
    on: +0.100 (standard error 0.028). arb: +0.030 (0.027). So of the 0.34 gap, about 0.1 is
    familiarity with the exact text, and about 0.3 is the step from the training sites to new sites
    (6.288 against 5.992). The two parts overlap and were not separated more finely.
- **Correction (D-022):** the ars *test* split, also made of new sites, scores 5.928 at step 1,500,
  only +0.10 above the train probe (5.830), against +0.31 for the validation split. So the
  "about 0.3 from the step to new sites" above is not a general cost of new sites: most of the
  validation gap belongs to the particular 39 validation sites, and what is left (about 0.1) is
  the size of the memorization estimate.
- **Reading.** Nothing is wrong with the training code. ars looks like a source with few, repetitive sites
  (its text was not read; this is inferred from the concentration of sites and from the
  seen/unseen result), which the model learns within a site quickly and which transfer to new
  sites less well; its validation set is also small and dominated by a few sites,
  so a single ars validation number is imprecise (about +-0.12 at 95%) and differences of a few
  hundredths between evaluations or mixes should not be read for that source. Validation is the
  honest number; the train probe is easier by construction (it shares sites with training).
- **Open.** A larger and more varied ars validation
  sample would need re-sampling the source (the sample is 3.4% of it). Reporting a site-level
  bootstrap interval with each evaluation, for the web sources, is a possible addition.

**Addendum (2026-11-03): the pilot was resumed and finished.** `train --resume auto` continued from
step 1,056 to step 1,500 (49,152,000 tokens, 0 skipped steps, exit 0; the log has one `interrupted`
and one `resume` event). The loss continued without a jump at the restart. Whether the resumed run
equals an uninterrupted one was not tested on this model; D-020 tested it on a small one.

| source | val loss @1000 | val loss @1500 | bits/char @1500 | train probe @1500 | gap @1500 |
|---|---:|---:|---:|---:|---:|
| edu | 4.61 | 4.44 | 1.91 | 4.41 | +0.03 |
| wiki_ar | 5.27 | 5.05 | 2.16 | 5.10 | -0.05 |
| arb | 5.65 | 5.47 | 2.22 | 5.51 | -0.04 |
| arz | 5.93 | 5.76 | 2.54 | 5.79 | -0.04 |
| ary | 5.86 | 5.68 | 2.24 | 5.65 | +0.03 |
| ars | 6.30 | 6.16 | 2.88 | 5.83 | +0.33 |
| **mean** | 5.60 | 5.43 | 2.33 | | |

- The mean validation loss fell every time it was measured, from 7.62 at step 100 to
  5.426 at step 1,500; the last three drops were 0.032, 0.025 and 0.017
  per 100 steps with the learning rate at its floor (6e-5). It had not plateaued.
- The ars gap is +0.33 at the end, as before; the other five sources are within 0.05 of their train
  probe (arb, arz and wiki_ar are slightly *lower* on validation).
- Peak GPU memory read 2,081 MiB after the resume against 2,012 MiB before; not investigated.
- **What this does and does not show.** A mean loss of 5.43 nats per token is far
  from good (a uniform guess is 9.70) and 49M tokens is 2.9 tokens per parameter, 0.29 of the
  167M-token train split. No text was generated (generation is not implemented), so nothing is known
  about what the model writes. The pilot shows that the whole path trains stably and reproducibly on
  the real data and fits the GPU; it does not show what the model can reach with more data or time.

## D-022 (2026-11-03): generation, full evaluation, and a first look at what the pilot model writes

Code: `model/generate.py`, `training/evaluation.py`, commands `scratchllm generate` and
`scratchllm eval`; report `reports/eval/final-pilot-17m` (numbers only, no text).

- **`generate`** writes one token at a time from the last position's scores: greedy (temperature
  0), or sampling from softmax(logits / temperature), optionally restricted to the top k tokens
  (ties with the k-th score are kept). Ending: a generated end-of-text id finishes a sequence
  (later steps pad it, and generation stops once all rows are done). Context: the model sees only
  its last `context_length` tokens; the output keeps everything. No key/value cache: each step
  recomputes the window (simple and correct, quadratic cost, fine for samples). `generate_text`
  normalizes the prompt (D-013), parses special-token strings as plain text, and starts the
  sequence with the end-of-text id, as every document starts in the training stream. The command
  refuses a tokenizer that is not the one the token files were made with (vocabulary size, and the
  sha256 stored in their descriptions when the files are still there).
- **Tests.** 25 for generation (a scripted fake model gives exact expectations: greedy order,
  top-k = 1 equals greedy, EOS handling for one row and for a batch, context never exceeded,
  sampled frequencies match softmax at three temperatures and after top-k within
  0.02, seeds) and 12 for evaluation and the tokenizer guard. Thirteen deliberate
  breakages (temperature unused, top-k off by one, EOS ignored, context not cropped, greedy
  picking the lowest, finished rows still sampling, prompt not normalized, no start token; loss
  summed instead of averaged, unsmoothed unigram, natural-log bits, eval-only treated as training,
  tokenizer hash unchecked) each fail a test.
- **Full evaluation** (every non-overlapping 512-token window of each split; 3 minutes on the
  GPU), checkpoint `best.pt` = step 1,500 of the pilot, 17,109,888 parameters:

| source | split | tokens | loss | perplexity | bits/char | unigram bits/char | model / unigram |
|---|---|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val | 1,548,223 | 4.450 | 86 | 1.914 | 3.330 | 0.57 |
| fineweb_edu_10bt | test | 1,380,127 | 4.472 | 88 | 1.924 | 3.321 | 0.58 |
| wiki_ar | val | 277,136 | 5.135 | 170 | 2.191 | 3.585 | 0.61 |
| wiki_ar | test | 319,077 | 5.261 | 193 | 2.211 | 3.532 | 0.63 |
| fw2_arb | val | 692,916 | 5.491 | 243 | 2.226 | 3.448 | 0.65 |
| fw2_arb | test | 855,059 | 5.552 | 258 | 2.276 | 3.475 | 0.65 |
| fw2_arz | val | 571,707 | 5.776 | 323 | 2.553 | 3.761 | 0.68 |
| fw2_arz | test | 729,537 | 5.741 | 311 | 2.580 | 3.829 | 0.67 |
| fw2_ary | val | 64,808 | 5.693 | 297 | 2.248 | 3.419 | 0.66 |
| fw2_ary | test | 203,565 | 5.611 | 273 | 2.085 | 3.272 | 0.64 |
| fw2_ars | val | 189,459 | 6.137 | 463 | 2.871 | 3.990 | 0.72 |
| fw2_ars | test | 159,996 | 5.928 | 375 | 2.917 | 4.158 | 0.70 |
| fw2_apc | eval | 4,472,543 | 6.065 | 430 | 2.949 | 4.120 | 0.72 |
| fw2_acm | eval | 5,470,966 | 6.289 | 539 | 3.212 | 4.331 | 0.74 |
| wiki_ary | eval | 2,232,171 | 6.477 | 650 | 3.664 | 4.695 | 0.78 |

  "unigram" is a model that knows only the token frequencies of the training data. Uniform guess:
  3.6 to 5.5 bits per character depending on the source.
- **Reading the numbers.**
  - The in-training estimate (80 windows per source) agrees with the full validation loss to
    within 0.083 nats on every source; the sample was adequate.
  - Test agrees with validation within 0.13 nats on every source except ars (test 5.928,
    validation 6.137); see the correction in D-021.
  - The model beats the unigram baseline on every split: bits per character are 0.66 of the
    unigram's on the five trained Arabic sources and 0.75 on the three dialects it never
    saw. It uses context, but a loss of 4.4 to 6.5 nats is a weak language model.
  - On the three evaluation-only dialects the cost is 3.28 bits per character against
    2.42 on the trained Arabic sources (mean over the sources, unweighted): the model and the
    tokenizer carry over to dialects they did not see, at a clear price. wiki_ary (Moroccan
    Wikipedia) is the hardest, 3.66.
  - Bits per character is the number to compare sources; per-token loss is not (different
    tokens per character).
- **What it writes** (6 prompts: Modern Standard, Egyptian, English; 90 tokens; temperature 0.8,
  top-k 40, two seeds each; and greedy; files kept in `runs/pilot-17m/`, not in git because
  generated web-like text has not been filtered for adult or personal content; the adult-content
  filter is still undecided):
  - Surface form is right in all three: correct script and spelling, the register of the prompt
    (news phrases such as "أعلنت وزارة الصحة", colloquial Egyptian "اللي / علشان كدة / هتلقي /
    بيعملوا", English sentences), punctuation and line structure, sometimes web boilerplate
    ("إقرأ أيضا", "Instagram").
  - Meaning is absent: sentences do not continue the prompt's topic, facts are invented (named
    officials and years attached to nonsense claims), English contains invented words ("Namentinia",
    "Swasse"), and a prompt can drift to another genre (an Egyptian prompt turned into a list of
    short chat lines).
  - Greedy decoding loops in all six cases ("اليوم الثلاثاء، اليوم الثلاثاء...", "the United States,
    the United States...", "النهار" on every line), as expected of a model this weak.
  - Six prompts and two seeds are an impression, not a measurement. Nothing here measures factual
    accuracy or dialect correctness.
- **Not decided.** Whether to train longer, and with what mix or settings (see the answer to the
  user for the evidence and recommendation; no longer run has been started).

## D-023 (2026-11-03): the 5,000-step baseline, and why the loss falls slowly: nothing broken was found

Code: none changed. Evidence kept in git: `reports/eval/final-local-17m` (full evaluation of the
final checkpoint) and `reports/diagnostics/final-d023/` (outputs of the diagnostic scripts).
Scripts: `scripts/diagnose_checkpoint.py`, `scripts/diagnose_tokens.py` and `scripts/overfit_one_batch.py`
made the files in `reports/diagnostics/final-d023/`; `scripts/d023_entry.py` computed the tables and numbers
of this entry from the runs and those files (the prose around them is written by hand). Not in git:
the runs (`runs/local-17m`, `runs/diag-lr-*`, ignored), so the entry can be regenerated only on the machine
that has them. Rerunning `diagnose_checkpoint.py` on the same checkpoint gives identical output except
the last digits of the noise-scale lines, which change from run to run (GPU kernels are not bitwise
deterministic).

**The baseline run.** `configs/local-17m.yaml`, unchanged from the pilot except for length:
17,109,888 parameters (6,291,456 in the token embedding, 196,608 in the position embedding, 10,621,824 elsewhere),
8 x 8 x 512 = 32,768 tokens per step,
natural mix, peak learning rate 0.0006 decaying to 6e-05. It was stopped by SIGTERM at step 2,990 (2026-11-03T05:07Z;
`last.pt` was written) and resumed from that checkpoint at 2026-11-03T16:17Z, then ran to step 5,000.
163,840,000 tokens were seen (0.98 epoch of the
167,394,059 train tokens), 0 steps were skipped, the median throughput was 34,424 tokens/s.

| step | tokens seen | mean val loss | mean bits/char |
|---:|---:|---:|---:|
| 500 | 16,384,000 | 6.166 | 2.640 |
| 1,000 | 32,768,000 | 5.566 | 2.385 |
| 1,500 | 49,152,000 | 5.243 | 2.247 |
| 2,000 | 65,536,000 | 4.962 | 2.127 |
| 2,500 | 81,920,000 | 4.810 | 2.062 |
| 3,000 | 98,304,000 | 4.703 | 2.016 |
| 3,500 | 114,688,000 | 4.631 | 1.986 |
| 4,000 | 131,072,000 | 4.579 | 1.963 |
| 4,500 | 147,456,000 | 4.545 | 1.949 |
| 5,000 | 163,840,000 | 4.524 | 1.940 |

Final per-source state (step 5,000); the probe is the loss on train windows, measured the same way as validation:

| source | share of train tokens | val loss | val bits/char | train-probe loss | val − probe |
|---|---:|---:|---:|---:|---:|
| edu | 39.3% | 3.634 | 1.563 | 3.581 | +0.053 |
| arb | 23.0% | 4.501 | 1.825 | 4.496 | +0.005 |
| ars | 8.0% | 5.300 | 2.479 | 4.925 | +0.375 |
| ary | 3.2% | 4.742 | 1.873 | 4.678 | +0.064 |
| arz | 17.7% | 4.800 | 2.121 | 4.653 | +0.146 |
| wiki_ar | 8.7% | 4.169 | 1.779 | 4.144 | +0.025 |

Mean validation loss 4.524 nats, 1.940 bits/char. Weighted by the
train mix the validation loss is 4.256, against 4.177 for the training loss (mean of the last five logged
windows): the same weighting, and a difference of 0.079, which is the generalization gap
in the usual sense; `ars` and `arz` (below) account for 68% of the train-mix weighted val-minus-probe gap of 0.082. The headline `val_loss` is an unweighted mean over six
sources, which is why it looks 0.35 higher than the training loss; the rest of that
difference (0.27) is the weighting, not a gap. The per-source
probe shows where the gap is: `ars` (+0.37, the concentrated
validation sites of D-021) and `arz` (+0.15, not investigated); the other four are within
0.06 nats.

**Full evaluation** of `best.pt` (step 5,000), every non-overlapping 512-token window of val, test and the
evaluation-only dialects; `change` is against the pilot (D-022):

| source | split | tokens | loss | perplexity | bits/char | pilot bits/char | change | unigram bits/char | model / unigram |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| edu | val | 1,548,223 | 3.619 | 37 | 1.557 | 1.914 | -0.357 | 3.330 | 0.47 |
| edu | test | 1,380,127 | 3.664 | 39 | 1.576 | 1.924 | -0.348 | 3.321 | 0.47 |
| acm | eval | 5,470,966 | 5.562 | 260 | 2.841 | 3.212 | -0.371 | 4.331 | 0.66 |
| apc | eval | 4,472,543 | 5.213 | 184 | 2.535 | 2.949 | -0.414 | 4.120 | 0.62 |
| arb | val | 692,916 | 4.530 | 93 | 1.836 | 2.226 | -0.390 | 3.448 | 0.53 |
| arb | test | 855,059 | 4.657 | 105 | 1.909 | 2.276 | -0.367 | 3.475 | 0.55 |
| ars | val | 189,459 | 5.332 | 207 | 2.494 | 2.871 | -0.377 | 3.990 | 0.63 |
| ars | test | 159,996 | 5.205 | 182 | 2.561 | 2.917 | -0.356 | 4.158 | 0.62 |
| ary | val | 64,808 | 4.742 | 115 | 1.873 | 2.248 | -0.376 | 3.419 | 0.55 |
| ary | test | 203,565 | 4.773 | 118 | 1.773 | 2.085 | -0.311 | 3.272 | 0.54 |
| arz | val | 571,707 | 4.785 | 120 | 2.115 | 2.553 | -0.438 | 3.761 | 0.56 |
| arz | test | 729,537 | 4.736 | 114 | 2.129 | 2.580 | -0.451 | 3.829 | 0.56 |
| wiki_ar | val | 277,136 | 4.135 | 62 | 1.764 | 2.191 | -0.427 | 3.585 | 0.49 |
| wiki_ar | test | 319,077 | 4.292 | 73 | 1.803 | 2.211 | -0.407 | 3.532 | 0.51 |
| wiki_ary | eval | 2,232,171 | 5.095 | 163 | 2.882 | 3.664 | -0.782 | 4.695 | 0.61 |

- Every split of a training source improved by 0.31 to 0.45 bits/char since the pilot (step 1,500), all of them by a similar amount.
- Mean of the five trained Arabic sources (val): 2.016; mean of the three dialects never seen: 2.753. The ratio is
  1.37; at the pilot it was 1.35 (3.275 / 2.418). The relative price of an unseen dialect did not move;
  what the model learned about the seen ones carried over in the same proportion.
- wiki_ary improved by 0.78, about twice the rest; it is the smallest of the three and the cause was not looked into.
- For one source, val and test differ by up to 0.10 bits/char (set against seed noise below): compare a split with the same
  split, never val with test.

**Why the loss falls slowly: what was checked and what it showed.** Each check is a measurement on the baseline or a short run
with one change.

- *Implementation.* The exact model, on CPU, memorizes four 128-token windows: loss 9.78 at the start (ln 16,384 = 9.70),
  0.0025 after 120 steps. Bf16 against fp32 on the validation windows differs by at most 0.001 nats on every source.
  The loop's own checks stand from D-020 (resume bit-exact on CPU, mutants caught). No defect found.
- *Gradient clipping.* The clip is 1.0. The log holds the norm of every 10th step: 3 of 500 logged steps were above it, the last at step 120
  (during warmup); after step 200 the median is 0.72 and the maximum 0.84. Clipping acted only at the very start, and it does not slow the run.
- *Learning rate.* Four runs of 800 steps, identical except the peak learning rate (minimum 10% of it, 50 warmup steps, seed 1337).
  Command, for the rate LR (the minimum is LR/10): `scratchllm train configs/local-17m.yaml --set name=diag-lr-X --set optim.lr=LR
  --set optim.min_lr=LR/10 --set optim.warmup_steps=50 --set train.max_steps=800 --set train.eval_interval=200
  --set train.checkpoint_interval=800`; the repeat adds `--set train.seed=2024`
  and `--set name=diag-lr-6e-4-seed2`:

| peak LR | val loss at 200 | 400 | 600 | 800 | mean bits/char at 800 | max grad norm after step 100 | skipped steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3e-4 | 7.129 | 6.701 | 6.494 | 6.409 | 2.743 | 0.64 | 0 |
| 6e-4 | 6.932 | 6.437 | 6.168 | 6.061 | 2.595 | 0.56 | 0 |
| 1.2e-3 | 6.991 | 6.559 | 6.293 | 6.162 | 2.639 | 0.52 | 0 |
| 2.4e-3 | 7.027 | 6.645 | 6.353 | 6.210 | 2.659 | 0.34 | 0 |

  6.061 at 6e-4 is the best; 3e-4 is 0.348 worse and 1.2e-3 is 0.102 worse, with no instability at any rate.
  A higher rate was the hypothesis going in (6e-4 is a value for much wider models) and it is refuted at this horizon. A short sweep tends to favour
  higher rates than a long run does, so the best rate at 5,000 steps is unlikely to be higher than 6e-4.
- *Noise.* The 6e-4 run repeated with seed 2024 (different initialization and data order, same validation windows) gives 6.089 against 6.061:
  +0.029 nats, +0.012 bits/char on the mean, +0.007 to +0.021 per source. One pair is a rough estimate of the noise; the
  learning-rate gaps above are 3.5 to 12.1 times it.
- *Batch size.* Gradient noise scale (McCandlish et al. 2018), 24 steps of 8 micro-batches, natural mix:
  B_noise = 75,130 tokens at step 2,750 (batch/B_noise = 0.44) and 223,148 at step 5,000 (0.15).
  It grows as the loss falls, as the theory says, so the batch of 32,768 tokens is further below it as training advances.
  Below the noise scale every token is used with efficiency 87% at step 5,000 (69% at step 2,750) of the best possible
  (E/E_min = 1 + B/B_noise); halving the batch would save about 7% of the tokens at step 5,000 and
  15% at step 2,750, with the learning rate to be retuned. The estimate is 24 steps at one checkpoint.
  The batch is not a cause of the slow fall.
- *Data.* Train split, from the token files: exact duplicate documents (within a source) 0.00% in every source; tokens in documents of fewer than 64 tokens 0.00%;
  the share of positions inside a 512-window whose previous 4-gram already occurred in it is 0.022 to 0.057 on average, and
  windows with more than 30% of them: 0.0% to 4.0% (wiki_ar, lists and tables).
- *Sources.* Cosine between the gradients of two 32,768-token batches, divided by what two batches of the same source reach (so 1 would mean as
  alike as the source is to itself). Step 5,000:

| | edu | arb | ars | ary | arz | wiki_ar |
|---|---:|---:|---:|---:|---:|---:|
| edu | 1 | 0.43 | 0.20 | 0.29 | 0.24 | 0.23 |
| arb | 0.43 | 1 | 0.23 | 0.47 | 0.45 | 0.53 |
| ars | 0.20 | 0.23 | 1 | 0.28 | 0.59 | 0.23 |
| ary | 0.29 | 0.47 | 0.28 | 1 | 0.35 | 0.40 |
| arz | 0.24 | 0.45 | 0.59 | 0.35 | 1 | 0.35 |
| wiki_ar | 0.23 | 0.53 | 0.23 | 0.40 | 0.35 | 1 |

  Step 2,750:

| | edu | arb | ars | ary | arz | wiki_ar |
|---|---:|---:|---:|---:|---:|---:|
| edu | 1 | 0.32 | 0.17 | 0.30 | 0.26 | 0.28 |
| arb | 0.32 | 1 | 0.34 | 0.75 | 0.59 | 0.77 |
| ars | 0.17 | 0.34 | 1 | 0.33 | 0.64 | 0.22 |
| ary | 0.30 | 0.75 | 0.33 | 1 | 0.57 | 0.64 |
| arz | 0.26 | 0.59 | 0.64 | 0.57 | 1 | 0.50 |
| wiki_ar | 0.28 | 0.77 | 0.22 | 0.64 | 0.50 | 1 |

  Raw cosines between different sources were never negative at either step (smallest 0.087 and 0.046): no
  sign of one source working against another. Arabic sources with each other are 0.50 to 0.77 at step 2,750 (arb, ary, arz,
  wiki_ar); English with Arabic 0.17 to 0.32; ars is close only to arz (the other Arabic sources
  0.22 to 0.34). At step 5,000 every value is smaller, because the
  gradient is mostly noise by then (the diagonal itself fell from 0.49 to 0.25), so the later matrix says less. Token
  statistics agree: the Jensen-Shannon divergence between unigram distributions is 0.04 to 0.17 among the Arabic sources
  and 0.57 to 0.62 between English and any of them (maximum 0.69); 1,551 ids are used by English only, 12,221 by Arabic only, 2,492 by both.
  Under 'is it appropriate to combine them': nothing found against it, and nothing measured for it either, see the open points.
- *Where the loss goes, by position.* On the validation windows the loss drops from the middle of the window (64-255) to the end (256-511) by only
  0.05 to 0.12 nats per source: the second half of the window adds little over the middle. The windows do not start at document starts, so
  this says how much context helps within a window, not how well long-range structure is understood.
- *Scale.* With the rounded Hoffmann et al. fit L = 1.69 + 406.4/N^0.34 + 410.7/D^0.28 (a different corpus and tokenizer, and the fit itself is disputed;
  read as an order of magnitude), N = 17.1M and D = 90M tokens (step 2,750) give 5.53 nats (5.78 counting only the 10.6M non-embedding parameters);
  the train-mix weighted validation loss at that step was 4.47, below it. Going on to 164M tokens the fit predicts a further fall of
  0.38 nats (0.35 with the unrounded exponents of the replication paper); the weighted validation loss fell by 0.22 (0.23 for the unweighted mean), so the fit overstates the
  further fall. It is for another corpus and tokenizer, so this is a limit of the fit, not a symptom. The run was announced before it finished as
  "final mean validation loss between 4.4 and 4.6" and ended at 4.524: inside the range, in its upper half. The model is in the range expected for its size and its budget of
  9.6 tokens per parameter (15.4 per non-embedding one; about 20 is the usual compute-optimal figure), not better or worse by a margin this analysis can resolve.

**Conclusion.** No implementation, optimizer, precision, batch or data fault was found. The loss falls as a 17M-parameter model on 164M tokens
is expected to fall. What makes it look worse than it is: the headline validation loss and the training loss are weighted differently; per-token
losses of different sources are not comparable (tokens per character range from 0.279 to 0.331); and the first
stop, at step 2,990, left the learning rate at 2.6e-04, in the middle of its decay, so the curve looked unfinished.

**Samples** (the same six prompts, settings and seeds as D-022; kept in `runs/local-17m/samples-*.txt`, not in git, for the same reason):
the first greedy sentence for the health-ministry prompt is now a plausible news sentence (a number of new deaths from a virus), where the
pilot's was not; greedy decoding still falls into a loop in all six prompts, after roughly 20 to 40 tokens. With sampling, Arabic keeps its
register and keeps drifting from the topic; web boilerplate appears ("تسجيل الدخول في وقت لاحق", a repeated "أبرزها", a chat-transcript speaker
label); the Egyptian prompts continue in a recognizable colloquial style; the English samples are the weakest, with invented or broken words
("jumpetal", "ented", "Self-Self-Self-") although English is the largest source. A possible reason is that the vocabulary was built for Arabic
and cuts English into more pieces (1.78 tokens per word against 1.32 for GPT-2, D-016); this was not tested. Six prompts and two seeds are an impression.

**Decisions.**

1. Hyperparameters, batch, clipping, precision and architecture of `local-17m` are kept; this run is the baseline every later change is compared with.
   Learning rate 6e-4 is not raised.
2. Bits per character is the number to compare runs and sources. Report it next to the unweighted mean loss and the train-mix weighted loss, and
   always on the same split.
3. Rule fixed before the next experiment, from the noise above: a change counts as an improvement only if the mean bits/char on the same split falls by
   at least 0.03 (about twice the 0.012 measured) and, for a claim about one source, by at least 0.04.

**Not decided, and not tested here.**

- Whether the English share (the largest, 39%) costs the Arabic sources anything. The gradient evidence says the sources do not fight, and
  it says English shares little with Arabic; neither says what removing it would do. It needs the mix ablation (natural, Arabic only, English at 20%;
  800 steps each, judged by rule 3), and it depends on the project wanting an English-capable model.
- Whether several epochs on the same 167M tokens, or more data, helps; the literature suggests repeated data is nearly as good as new for a few epochs (recalled, not checked in this session).
- The English tokenization hypothesis above; the `arz` train/validation gap; the single-seed basis of every run except the one repeated.
- README still shows the pilot results; it has not been updated.

## D-024 (2026-11-05): the English share, 39% to 20% of the tokens: Arabic gains 0.044 bits/char at 5,000 steps, English loses 0.074

Code: none changed. Added: `configs/ablation-arabic-only.yaml`, `configs/ablation-english-20.yaml` (800 steps) and
`configs/ablation-english-20-long.yaml` (5,000 steps), each with the question and the rule of D-023 in its header, written
before the run. Evidence kept in git: `reports/eval/final-ablation-english-20-long/` (full evaluation of the 5,000-step
checkpoint) and `reports/eval/final-d024-mix-ablation/evals.json` (the evaluation events of the six runs below, copied from
their `metrics.jsonl`). Script: `scripts/d024_entry.py` computed the tables and numbers of this entry from those runs and the
two full evaluations (prose by hand; sentences that depend on a comparison are guarded by an assert in the script). Not in git:
the runs themselves (`runs/`, ignored). The first attempt of the Arabic-only arm was killed when the session restarted, at step 370,
and rerun from step 0; its directory was kept aside and is not used.

**The question** (left open in D-023): does the English share, the largest source at 39.3% of the training tokens, cost the Arabic sources
anything? D-023's gradient evidence said the sources do not fight and share little; it could not say what removing English would do.
**The rule** (D-023, decision 3, fixed before the runs): a change counts only if the mean bits/char of the five Arabic sources on the same
split falls by at least 0.03, and a claim about one source needs at least 0.04. English is reported and is not part of the rule.

Everything else is identical to `configs/local-17m.yaml`: the model, optimizer, batch (32,768 tokens), seed 1337 and the validation
windows. The Arabic sources keep their natural relative weights; only the English weight changes. Training batches are random windows drawn
with replacement, so "passes" below are expected coverage (tokens drawn from a source / its tokens in the files).

| run | steps | English share of train tokens | Arabic share | tokens seen | expected passes over the Arabic data | over the English data |
|---|---:|---:|---:|---:|---:|---:|
| control (`diag-lr-6e-4`) | 800 | 39.3% | 60.7% | 26,214,400 | 0.16 | 0.16 |
| `ablation-arabic-only` | 800 | 0.0% | 100.0% | 26,214,400 | 0.26 | 0.00 |
| `ablation-english-20` | 800 | 20.0% | 80.0% | 26,214,400 | 0.21 | 0.08 |
| baseline (`local-17m`) | 5,000 | 39.3% | 60.7% | 163,840,000 | 0.98 | 0.98 |
| `ablation-english-20-long` | 5,000 | 20.0% | 80.0% | 163,840,000 | 1.29 | 0.50 |

**800 steps** (the control is the learning-rate run `diag-lr-6e-4`, same settings, 50 warmup steps; bits/char, validation, step 800).
Seed noise measured on this control and its repeat (seed 2024): +0.013 on the Arabic mean and +0.010 to +0.021 per source
(D-023 quoted 0.012, the mean over six sources with English; the Arabic mean used by the rule is 0.013).

| run (step 800) | arb | ars | ary | arz | wiki_ar | Arabic mean | change vs control | English | change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 2.479 | 3.095 | 2.506 | 2.825 | 2.518 | 2.685 |  | 2.147 |  |
| Arabic only | 2.388 | 3.002 | 2.398 | 2.716 | 2.399 | 2.581 | -0.104 | 2.892 | +0.744 |
| English 20% | 2.450 | 3.062 | 2.468 | 2.790 | 2.479 | 2.650 | -0.035 | 2.282 | +0.135 |

Arabic only: mean -0.104 (mean meets the 0.03 threshold; 5 of 5 sources reach 0.04); English +0.744.
English 20%: mean -0.035 (mean meets the 0.03 threshold; 0 of 5 sources reach 0.04); English +0.135.
Both pass on the mean; the 20% arm passes only there, since no single source reaches 0.04 (its gain per source is 0.029 to 0.038, against seed noise of up to 0.021 per source).
The Arabic-only arm gains about 3.0 times as much as the 20% arm, at the cost of English losing 0.74 bits/char (English was not in its mix at all).
Only 800 steps, one seed per arm, and the English share and the Arabic exposure change together.

**5,000 steps** (English 20% against the baseline of D-023, `local-17m`; the in-training validation evaluation, 20 batches per source):

| run (step 5,000, validation) | arb | ars | ary | arz | wiki_ar | Arabic mean | English |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline, English 39.3% | 1.825 | 2.479 | 1.873 | 2.121 | 1.779 | 2.0153 | 1.5634 |
| English 20.0% | 1.781 | 2.438 | 1.826 | 2.075 | 1.737 | 1.9715 | 1.6375 |
| change | -0.044 | -0.041 | -0.047 | -0.046 | -0.042 | -0.0438 | +0.0741 |

By the rule: mean meets the 0.03 threshold; 5 of 5 sources reach 0.04. The margin per source is thin: the smallest gain is 0.041, against a threshold of 0.04 and a per-source seed noise of up to 0.021 measured at 800 steps; the mean gain is
0.0438, 1.5 times the threshold and 3.3 times the seed noise measured at 800 steps. English is +0.0741, not part of the rule.
Caveat on the baseline: it was stopped once at step 2,990 and resumed; the 20% run was not interrupted.

The gap is there at every checkpoint, not only the last:

| step | baseline Arabic mean | English 20% Arabic mean | change | baseline English | English 20% English | change |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 2.729 | 2.685 | -0.044 | 2.195 | 2.299 | +0.105 |
| 1,000 | 2.465 | 2.417 | -0.048 | 1.986 | 2.075 | +0.090 |
| 1,500 | 2.328 | 2.276 | -0.053 | 1.842 | 1.944 | +0.102 |
| 2,000 | 2.208 | 2.157 | -0.051 | 1.723 | 1.810 | +0.087 |
| 2,500 | 2.141 | 2.089 | -0.052 | 1.666 | 1.748 | +0.082 |
| 3,000 | 2.094 | 2.047 | -0.047 | 1.627 | 1.706 | +0.079 |
| 3,500 | 2.062 | 2.016 | -0.046 | 1.601 | 1.679 | +0.078 |
| 4,000 | 2.039 | 1.994 | -0.045 | 1.583 | 1.658 | +0.075 |
| 4,500 | 2.024 | 1.980 | -0.045 | 1.571 | 1.645 | +0.074 |
| 5,000 | 2.015 | 1.972 | -0.044 | 1.563 | 1.637 | +0.074 |

It reaches 0.03 already at step 500 and stays beyond it at every checkpoint: -0.044 at step 500, -0.053 at its largest (step 1,500) and -0.044 at 5,000, so the 800-step result carried over to the full length and settled a little lower as the loss flattened.
The English loss is +0.105 at step 500, +0.082 at 2,500 and +0.074 at 5,000.

**Full evaluation** of the two final checkpoints (every held-out token; `reports/eval/final-local-17m` and `reports/eval/final-ablation-english-20-long`).
Its validation numbers differ slightly from the table above because it reads every validation token, not 20 batches; the comparison is between like and like.
The test split and the three dialects never seen in training are not part of the rule; they are the check that the validation result is not an accident of the validation windows.

| source | split | baseline bits/char | English 20% | change |
|---|---|---:|---:|---:|
| edu | val | 1.557 | 1.632 | +0.076 |
| edu | test | 1.576 | 1.651 | +0.075 |
| arb | val | 1.836 | 1.793 | -0.044 |
| arb | test | 1.909 | 1.867 | -0.042 |
| ars | val | 2.494 | 2.455 | -0.039 |
| ars | test | 2.561 | 2.523 | -0.038 |
| ary | val | 1.873 | 1.826 | -0.047 |
| ary | test | 1.773 | 1.732 | -0.041 |
| arz | val | 2.115 | 2.068 | -0.047 |
| arz | test | 2.129 | 2.080 | -0.048 |
| wiki_ar | val | 1.764 | 1.722 | -0.042 |
| wiki_ar | test | 1.803 | 1.760 | -0.044 |
| acm | eval | 2.841 | 2.814 | -0.028 |
| apc | eval | 2.535 | 2.494 | -0.040 |
| wiki_ary | eval | 2.882 | 2.874 | -0.008 |
| **trained Arabic, val (mean)** | | **2.016** | **1.973** | **-0.044** |
| **trained Arabic, test (mean)** | | **2.035** | **1.992** | **-0.043** |
| **unseen dialects (mean)** | | **2.753** | **2.727** | **-0.025** |

The change on the Arabic mean is -0.044 on validation and -0.043 on test, both beyond 0.03. Per source, 4 of 5 reach 0.04 on validation and 4 of 5 on test; ars falls just short on at least one of them,
so a claim about every single source (the 0.04 rule) holds on the in-training validation table above and is thinner on the full evaluation. On the 3 dialects never seen in training the sign is the same for all three, but the mean change is -0.025, under 0.03, and wiki_ary moves by only -0.008:
read it as no worse on unseen dialects, not as better.

**What the 20% run does not rule out.** At 20%, the Arabic data is drawn 1.29 times in expectation over 5,000 steps, against 0.98 for the baseline. Part of the
gain may be exposure to Arabic rather than relief from English (the same run cannot tell the two apart), and if repeating data hurt, it would show as a growing gap between the loss on held-out data and on the training data. The held-out loss minus the loss on a probe of
the training data (same tokens, so comparable across runs), in nats:

| run | arb | ars | ary | arz | wiki_ar | mean |
|---|---:|---:|---:|---:|---:|---:|
| baseline | +0.005 | +0.375 | +0.064 | +0.146 | +0.025 | +0.123 |
| English 20% | +0.009 | +0.390 | +0.082 | +0.180 | +0.030 | +0.138 |

The gap is larger in the 20% run for 5 of 5 Arabic sources (arb, ars, ary, arz, wiki_ar) by 0.005 to 0.034 nats, so
the 20% run fits its training data a little more tightly relative to held-out data; held-out loss still fell, which is what the rule measures, and at 1.29 passes this is not yet overfitting by any large amount. Besides
this, every run here has one seed (noise 0.013 on the Arabic mean, measured once, at 800 steps), the baseline was interrupted once, and the English tokenization
and the `arz` train/validation gap of D-023 are still untested.

**Decisions.**

1. The English-20% mix meets the rule of D-023 at the full length (mean 0.044 against 0.03; all five sources beyond 0.04 by a thin margin), and the mean gain holds on the test split. This is recorded
   as the result; the question "does English cost the Arabic sources anything" has the answer: at this model size and length, cutting English from 39% to 20% of the tokens
   improved the Arabic sources by 0.044 bits/char, and English paid 0.074.
2. `configs/local-17m.yaml` is not changed, and no checkpoint is replaced. Whether to adopt the 20% mix (or another) as the default is a project decision, taken on
   how much English the model must keep, not an experimental one: the experiment gives the exchange rate, English losing about 1.7 times what the Arabic mean gains.
3. The rule of D-023 stands unchanged.

**Not decided, and not tested here.**

- Whether the mechanism is less English or more Arabic exposure. A run that gives the Arabic data the same 1.29 passes while keeping English at 39% (about 216M tokens, 6,593 steps)
  would separate the two; it is also the test of whether more steps alone help.
- The Arabic-only arm at 5,000 steps (not run; 800 steps only), other English shares (10%, 30%), and a second seed of the 5,000-step pair. Each result above
  is a single seed.
- How much English the project wants the model to keep (D-023's open point, now with a price), and whether English quality matters more than its bits/char
  number says; the English samples were already the weakest in D-023.
