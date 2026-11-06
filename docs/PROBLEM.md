# Problem, solution and evaluation

## Problem

1. **Educational gap.** The best-known from-scratch LLM resources are English-only and
   notebook-first. They do not address Arabic, and they stop short of the engineering that
   makes a result reproducible (tests, typed configs, resumable training, CI).
2. **Arabic tokenization cost.** English-centric tokenizers split Arabic into many more tokens
   per word. Arabic is morphologically rich and has surface variation that carries little
   meaning (diacritics, tatweel, alef/yeh/teh-marbuta variants). More tokens means longer
   sequences, higher training cost and weaker models at a fixed budget.

## Solution

A tested, reproducible pipeline for a small bilingual GPT. The list is the plan; the **Status**
column says what exists, and the decision log entries hold the evidence.

| # | Planned | Status |
|---|---------|--------|
| 1 | Arabic-aware normalization + a bilingual BPE tokenizer trained on our own data | **Built** (D-013 to D-016). Normalization is deliberately light: it keeps spelling variants (D-006). Fertility is measured against GPT-2 and Qwen3. |
| 2 | A modern decoder-only transformer (RMSNorm, RoPE, GQA, SwiGLU, KV cache) written from scratch | **Mostly.** The default is still the GPT-2 architecture of the book (D-018, D-019); RMSNorm, rotary embeddings and SwiGLU are built as config switches (D-027) and, all three together scored lower than it at 5,000 steps on the trained languages in one run, not on the unseen Moroccan Wikipedia; a second seed is pending (D-028). GQA and a KV cache are **not built**; each further change is an ablation judged by bits per character (D-007, D-023), not a given. |
| 3 | Resumable pretraining (mixed precision, gradient accumulation, checkpointing) | **Built** (D-020): bfloat16, accumulation, clipping, warmup and cosine schedule, atomic checkpoints, exact resume, JSON-lines log. Only a 1,500-step pilot has been trained (D-021). |
| 4 | LoRA/SFT to turn the base model into a chat model | **Not built.** `finetune/` is empty. |
| 5 | Typed YAML configs, unit tests and CI so every number in this repo can be reproduced | **Configs and tests built** (typed, unknown keys refused; over 500 tests, mutation-checked). **CI exists but has not run on this code:** the workflow triggers on pushes to `main` while the branch is `master`, it installs the CUDA build of torch, and nothing has been pushed yet. |

Also built, as the plan required before any training: the data pipeline (sampling with manifests,
deterministic split by site, exploratory checks, cleaning rules, personal-data masking, leakage
measurement; D-001 to D-012, D-017) and generation (D-022).

The goal is not to compete with large Arabic LLMs. The goal is a clear, measurable and honest
engineering result at a scale that fits one 4 GB GPU, with larger configs ready for rented compute.

## Evaluation

**Primary metrics** (reported in the README for every released tokenizer/model):

| Metric | Definition | Compared against |
|--------|-----------|------------------|
| Tokenizer fertility | Mean tokens per whitespace word, per language, on held-out text | GPT-2 tokenizer, and any other public tokenizer whose license permits it |
| Perplexity | Exp of mean cross-entropy on held-out Arabic and English sets, reported per language | Our own baselines (ablations), not other models, unless tokenizers match |

Perplexity is not comparable across different tokenizers, because it is per token. Cross-tokenizer
comparisons must use bits-per-character (or bits-per-byte) instead.

**Supporting metrics** (not optional; they keep the primary metrics honest):

- Bits-per-character on held-out text, for cross-tokenizer comparison.
- Tokenizer round-trip: `decode(encode(x))` equals the normalized `x`, with 100% pass rate on the test set.
- Training throughput (tokens/sec) and peak VRAM on the target GPU.
- Qualitative samples in both languages at fixed prompts and seeds, with failures kept in the report.
- For SFT: a small hand-written instruction set scored by rubric, with the limits stated.
- Reproducibility: a run from a clean checkout, config and seed gives the same loss curve within tolerance.

## Where each metric stands

| Metric | State |
|---|---|
| Tokenizer fertility vs GPT-2 (and Qwen3) | Measured, `reports/tokenizer/final-v2` (D-016, D-017). Arabic far better than GPT-2; English worse. |
| Perplexity per language | Measured per source and split for the 5,000-step baseline, `reports/eval/final-local-17m` (D-023), and for the pilot, `reports/eval/final-pilot-17m` (D-022). English and Arabic are separate sources; there is no single "per language" figure yet. |
| Bits per character | Measured with the same report, with unigram and uniform baselines; the fair unigram baseline counts one language only (D-026, `reports/eval/final-d026-unigram`). |
| Tokenizer round trip | Tested, and 0 failures on 2,700 real documents (D-015). |
| Throughput and peak VRAM | Measured on the RTX 3050 (D-019, D-020). |
| Qualitative samples | Six prompts in Modern Standard, Egyptian and English (D-022, D-023): fluent surface; after 5,000 steps a first sentence can make sense, but there is no coherence over several sentences and greedy decoding loops. |
| SFT rubric | Not applicable yet. |
| Reproducibility | A resumed run is bit-identical to an uninterrupted one on CPU (tested) and in the one GPU trial made (D-020). A clean-checkout reproduction of a full run has not been done. |

## Rules for credible claims

- The evaluation sets are separated from training data before any cleaning, and checked for overlap.
- Every reported number names its config, seed, data version and commit.
- Negative and null results stay in the report.
