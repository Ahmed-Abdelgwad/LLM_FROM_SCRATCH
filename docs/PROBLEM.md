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

A tested, reproducible pipeline for a small bilingual GPT:

1. Arabic-aware normalization + a bilingual BPE tokenizer trained on our own data.
2. A modern decoder-only transformer (RMSNorm, RoPE, GQA, SwiGLU, KV cache) written from scratch.
3. Resumable pretraining (mixed precision, gradient accumulation, checkpointing).
4. LoRA/SFT to turn the base model into a chat model.
5. Typed YAML configs, unit tests and CI so every number in this repo can be reproduced.

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

## Rules for credible claims

- The evaluation sets are separated from training data before any cleaning, and checked for overlap.
- Every reported number names its config, seed, data version and commit.
- Negative and null results stay in the report.
