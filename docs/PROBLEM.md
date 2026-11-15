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
| 2 | A modern decoder-only transformer (RMSNorm, RoPE, GQA, SwiGLU, KV cache) written from scratch | **Built.** The default is still the GPT-2 architecture of the book (D-018, D-019); RMSNorm, rotary embeddings and SwiGLU are built as config switches (D-027) and, all three together scored lower than it at 5,000 steps on the trained languages in two seeds, not on the unseen Moroccan Wikipedia (D-028, D-029). The reference architecture is rotary embeddings + SwiGLU + LayerNorm, `configs/reference-17m.yaml` (RMSNorm was tested and dropped, D-035, D-036); a KV cache is built for generation (D-030); grouped-query attention is built as a switch, `model.n_kv_heads`, and was not adopted: with 2 key/value heads it cost +0.033 and +0.005 bits/char on the Arabic mean at the two seeds, failing its registered criterion (D-044, D-045); each further change is an ablation judged by bits per character (D-007, D-023), not a given. |
| 3 | Resumable pretraining (mixed precision, gradient accumulation, checkpointing) | **Built** (D-020): bfloat16, accumulation, clipping, warmup and cosine schedule, atomic checkpoints, exact resume, JSON-lines log. Only a 1,500-step pilot has been trained (D-021). |
| 4 | LoRA/SFT to turn the base model into a chat model | **Built** (D-038 to D-040): full fine-tuning and LoRA from scratch, on Aya and CIDAR. Replies come in the prompt's language and often stop, but most answers are wrong at this size; full fine-tuning answers better and forgets more of the pretraining, LoRA the reverse. Neither passes all four registered criteria. Replay of pretraining text in 20% of the fine-tuning rows, at the same steps and tokens, removes more than half of full fine-tuning's forgetting with no measured cost to the answers (D-042, D-043). |
| 5 | Typed YAML configs, unit tests and CI so every number in this repo can be reproduced | **Configs and tests built** (typed, unknown keys refused; 661 tests, mutation-checked). **CI is written and has not run on GitHub**, because nothing has been pushed. Its steps (ruff, format check, mypy, pytest) pass on an export of the committed files, run with the GPU hidden, so no test needs `data/`, `runs/` or a GPU. It installs the CUDA build of torch, which makes the install slow; a CPU-only index is a TODO in the workflow. |

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
| SFT rubric | Built, not scored. A blind sheet of 30 test-split prompts answered by the base, fully fine-tuned and LoRA models (InstructGPT's 1-to-7 overall score and two labels) is in `reports/sft/final-rubric-v1` (D-041); the scoring is a person's to do. Automatic measures (response bits per character, stop rate, language, looping) and samples are in `reports/sft/final-sft-*` (D-040). |
| Reproducibility | A resumed run is bit-identical to an uninterrupted one on CPU (tested) and in the one GPU trial made (D-020). Two GPU runs of the same code and config are not bit-identical: fine-tuning losses at step 10 differ in the fifth decimal, and final numbers by up to 0.008 bits/char (D-043). A clean-checkout reproduction of a full run has not been done. |

## Rules for credible claims

- The evaluation sets are separated from training data before any cleaning, and checked for overlap.
- Every reported number names its config, seed, data version and commit.
- Negative and null results stay in the report.
