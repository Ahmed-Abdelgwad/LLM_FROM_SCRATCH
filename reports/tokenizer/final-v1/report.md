# Tokenizer fertility

Tokenizer file `artifacts/tokenizer/bpe-32k.json` (sha256 `698bb54137491734`). Up to 600 held-out documents per source, cleaned and normalized, identical for every tokenizer.

- *tokens per word*: tokens / `str.split()` words. Lower is better. Arabic words carry glued prefixes, so compare tokenizers with each other, not Arabic with English.
- *characters per token*: independent of the word definition. Higher is better.
- Sources marked *evaluation-only* were never used to train the tokenizer.
- Our special tokens (`<email>`, `<phone>`) count as one token; outside tokenizers see them as text. The count of ours is in the last table.

## Tokens per word

| source | role | ours-8192 | ours-16384 | ours-24576 | ours-32768 | openai-community/gpt2 | Qwen/Qwen3-4B-Instruct-2507 |
|---|---|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val+test | 2.03 | 1.78 | 1.67 | 1.60 | 1.32 | 1.31 |
| fw2_acm | evaluation-only (never seen) | 2.07 | 1.89 | 1.79 | 1.73 | 5.31 | 2.52 |
| fw2_apc | evaluation-only (never seen) | 1.98 | 1.78 | 1.69 | 1.62 | 5.38 | 2.36 |
| fw2_arb | val+test | 1.87 | 1.66 | 1.56 | 1.49 | 5.80 | 2.18 |
| fw2_ars | val+test | 2.02 | 1.82 | 1.72 | 1.66 | 5.43 | 2.40 |
| fw2_ary | val+test | 1.84 | 1.63 | 1.54 | 1.48 | 6.14 | 2.44 |
| fw2_arz | val+test | 1.88 | 1.69 | 1.59 | 1.53 | 5.28 | 2.27 |
| wiki_ar | val+test | 1.96 | 1.76 | 1.66 | 1.60 | 5.83 | 2.38 |
| wiki_ary | evaluation-only (never seen) | 2.13 | 2.01 | 1.91 | 1.87 | 4.91 | 2.59 |

## Characters per token

| source | role | ours-8192 | ours-16384 | ours-24576 | ours-32768 | openai-community/gpt2 | Qwen/Qwen3-4B-Instruct-2507 |
|---|---|---:|---:|---:|---:|---:|---:|
| fineweb_edu_10bt | val+test | 2.92 | 3.33 | 3.57 | 3.72 | 4.50 | 4.55 |
| fw2_acm | evaluation-only (never seen) | 2.58 | 2.83 | 2.98 | 3.09 | 1.01 | 2.12 |
| fw2_apc | evaluation-only (never seen) | 2.71 | 3.01 | 3.18 | 3.30 | 1.00 | 2.27 |
| fw2_arb | val+test | 3.16 | 3.56 | 3.79 | 3.95 | 1.02 | 2.70 |
| fw2_ars | val+test | 2.71 | 3.02 | 3.19 | 3.31 | 1.01 | 2.29 |
| fw2_ary | val+test | 3.40 | 3.83 | 4.07 | 4.24 | 1.02 | 2.56 |
| fw2_arz | val+test | 2.84 | 3.17 | 3.36 | 3.49 | 1.01 | 2.36 |
| wiki_ar | val+test | 3.04 | 3.40 | 3.60 | 3.74 | 1.03 | 2.51 |
| wiki_ary | evaluation-only (never seen) | 2.41 | 2.55 | 2.69 | 2.74 | 1.04 | 1.98 |

## Mean over sources of each language (unweighted)

| tokenizer | vocab | Arabic tokens/word | Arabic chars/token | English tokens/word | English chars/token |
|---|---:|---:|---:|---:|---:|
| ours-8192 | 8192 | 1.97 | 2.86 | 2.03 | 2.92 |
| ours-16384 | 16384 | 1.78 | 3.17 | 1.78 | 3.33 |
| ours-24576 | 24576 | 1.68 | 3.36 | 1.67 | 3.57 |
| ours-32768 | 32768 | 1.62 | 3.48 | 1.60 | 3.72 |
| openai-community/gpt2 | 50257 | 5.51 | 1.02 | 1.32 | 4.50 |
| Qwen/Qwen3-4B-Instruct-2507 | 151669 | 2.39 | 2.35 | 1.31 | 4.55 |

## Our special tokens in the evaluation text

130 special tokens out of 4,579,792 tokens (0.0028%) for `ours-8192`.
