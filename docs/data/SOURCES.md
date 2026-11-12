# Data sources

Status: **candidates, sizes verified from the Hugging Face datasets-server on 2026-10-24.**
Nothing is downloaded yet. A source is "approved" only after the license check below is
re-read from the dataset card at download time and the snapshot (revision hash) is recorded.

Scope decision: open, ready-made sources; Arabic includes dialects (not only MSA).

## Arabic

### FineWeb-2 (`HuggingFaceFW/fineweb-2`), primary web source
- License on the card: **ODC-By**. The underlying text comes from Common Crawl, so the
  Common Crawl terms of use also apply, and the text may contain personal data.
- Row fields: `text, id, dump, url, date, file_path, language, language_score,
  language_script, minhash_cluster_size, top_langs, wordlist_ratio`.
- Each config has `train` and `test` splits.

| Config | Variety | Train docs | Train size (parquet) |
|--------|---------|-----------:|---------------------:|
| `arb_Arab` | Standard Arabic | 61,977,525 | ~106 GB |
| `arz_Arab` | Egyptian | 853,290 | ~1.1 GB |
| `ary_Arab` | Moroccan | 2,365,405 | ~2.9 GB |
| `ars_Arab` | Najdi | 298,167 | ~0.8 GB |
| `apc_Arab` | North Levantine | 69,740 | ~65 MB |
| `acm_Arab` | Mesopotamian | 7,382 | ~11 MB |

Plan: **stream and sample**; do not download `arb_Arab` in full (free disk is ~43 GB).
Caveat to test in the EDA: the dialect configs are assigned by a language-ID model, so labels
are noisy, and `arb_Arab` itself contains dialectal text. Do not treat the labels as ground truth.

### Wikipedia (`wikimedia/wikipedia`, dump `20231101`)
- License: **CC BY-SA 3.0 and GFDL**. Share-alike applies to derived text; keep this in mind
  before redistributing any processed corpus (redistributing trained weights is a separate,
  unsettled question; record the decision in `docs/DECISIONS.md`).
- Configs: `20231101.ar` (1.22M rows), `20231101.arz` (1.62M rows), `20231101.ary` (8.09k rows).
- Use: clean, encyclopedic reference text, and a good held-out evaluation source.

## English

### FineWeb-Edu (`HuggingFaceFW/fineweb-edu`)
- License: **ODC-By**. Subsets `sample-10BT` (~9.67M rows), `sample-100BT`, `sample-350BT`.
- Plan: stream a slice of `sample-10BT`; we need hundreds of millions of tokens, not 10B.
- Has an educational-quality `score`, which we can use as a filter and report on.

### Wikipedia `20231101.en` (6.41M rows), same license as above.

## Considered and not chosen (yet)

- **CulturaX** (`uonlp/CulturaX`): gated; terms "strictly follow" mC4 and OSCAR, so licensing
  is two layers deep. Arabic has 74.0M docs and 69.4B tokens. It overlaps in origin with
  FineWeb-2, so adding it mostly adds license complexity. Revisit only if we need more volume.

## Fine-tuning (instruction) data (D-038)

Pinned and hashed by `scratchllm data sft-prepare`; files under `data/raw/sft/`, splits under `data/processed/sft/`.

| Source | Revision | Used | License | Notes |
|---|---|---|---|---|
| Aya Dataset (`CohereLabs/aya_dataset`) | `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` | Standard Arabic, Egyptian, English; train and test files | Apache 2.0 | Human-written. The Moroccan subset is not used (D-038). |
| CIDAR (`arbml/CIDAR`) | `bc2f9d7a9de34b534b126c32b8cce98e098138df` | all 10,000 pairs | Apache 2.0 on the card | Translated from Stanford Alpaca (CC BY-NC 4.0, answers by an OpenAI model) with ChatGPT, plus 891 pairs from an Al Jazeera site: fine for this non-commercial project; check again before any commercial use. |

## Held-out evaluation data

Carve out validation and test sets from each source **before any cleaning or deduplication**,
split by document (and by URL domain for web data, to reduce near-duplicate leakage). Record
the split seed. FineWeb-2's own `test` splits are an additional independent check.

## Provenance log (fill in at download time)

| Source | Config/subset | HF revision | Docs | SHA-256 of shard list | Date |
|--------|---------------|-------------|-----:|-----------------------|------|
| | | | | | |
