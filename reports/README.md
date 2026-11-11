# reports/

Only reports named `final-*` are tracked in git; every other folder here is a working run and is ignored (`.gitignore`).
Nothing in this tree holds raw text or checkpoints. Folder names are cited by path in `docs/DECISIONS.md`, in the README
and in `scripts/archive/`, so they are never renamed or merged: a name below is the name the log uses.

## What each kind of folder holds

| Kind | Files | Written by |
|---|---|---|
| `eval/final-<run>/` | `report.json`, `report.md`: the full evaluation of one checkpoint, bits per character per source and split, with the uniform and unigram baselines | `scratchllm eval --name final-<run>` |
| `eval/final-dNNN-*/` | `evals.json`: the numbers a decision entry compares, copied from the runs' `metrics.jsonl` (which git does not hold). Keyed by label, then the run's name and its evaluation events (D-024: its end values and weights instead) | the entry script of that decision, in `scripts/archive/` |
| `eval/final-d026-unigram/` | `report.json`, `report.md`: the unigram baseline counted within one language | `scripts/archive/d026_unigram.py` |
| `diagnostics/final-dNNN/` | plain-text output of the diagnostic scripts; the header of each file says what it was run on | `scripts/diagnose_checkpoint.py`, `scripts/diagnose_tokens.py`, `scripts/overfit_one_batch.py`, `scripts/check_kv_cache.py` |
| `eda/final-*/` | `report.json`, `report.md` (no raw text) | `scratchllm data analyze`, `scratchllm data leakage-report` |
| `tokenizer/final-*/` | fertility measurements of the tokenizers | `scratchllm tokenizer fertility` |

## Which decision uses which folder

| Folder | Entry | Content |
|---|---|---|
| `eda/final-v1` | D-001 and the entries after it | first data analysis; kept although `final-v2` replaces it |
| `eda/final-v2` | D-011 | analysis corrected for placeholder e-mails |
| `eda/final-leakage-v1` | D-012 | train/evaluation leakage across all sources |
| `tokenizer/final-v1` | D-016 | fertility against GPT-2 and Qwen3 |
| `tokenizer/final-v2` | D-017 | the 16,384-id tokenizer |
| `eval/final-pilot-17m` | D-022 | the 49M-token pilot |
| `eval/final-local-17m` | D-023 | the baseline of record: 5,000 steps, 164M tokens |
| `diagnostics/final-d023` | D-023 | overfit-one-batch, checkpoint and token-file checks |
| `eval/final-ablation-english-20-long`, `eval/final-d024-mix-ablation` | D-024 | the data-mix ablation |
| `eval/final-d026-unigram` | D-026 | the same-language unigram baseline |
| `eval/final-modern-17m`, `eval/final-d028-modern-arch` | D-028 | the first modern package, seed 1337 |
| `eval/final-local-17m-seed2024`, `eval/final-modern-17m-seed2024`, `eval/final-d029-seed-replicate` | D-029 | the second seed of both |
| `diagnostics/final-d030` | D-030 | KV-cache check on real checkpoints |
| `eval/final-ablation-rmsnorm-only`, `eval/final-d032-ablation-stage1` | D-032 | first stage of the component ablation |
| `eval/final-ablation-swiglu-only`, `eval/final-ablation-swiglu-only-seed2024`, `eval/final-d033-swiglu-seeds` | D-033 | SwiGLU alone, two seeds |
| `eval/final-ablation-rope-only`, `eval/final-ablation-rope-only-seed2024`, `eval/final-ablation-rope-swiglu`, `eval/final-d035-followups` | D-035 | rotary alone at two seeds, the package without RMSNorm at seed 1337 |
| `eval/final-ablation-rope-swiglu-seed2024`, `eval/final-d036-no-rmsnorm` | D-036 | the package without RMSNorm at seed 2024; basis of `reference-17m` |

## Regenerating

The `evals.json` files and the entries of `docs/DECISIONS.md` are rebuilt by `scripts/archive/dNNN_entry.py`. They need
`runs/`, which is not in git, so on a fresh clone they document how a number was computed but cannot be rerun there. The
full evaluations need the checkpoint of the run and the token files under `data/`.
