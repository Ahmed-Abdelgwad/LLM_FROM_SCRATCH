# scratchllm

A bilingual (Arabic/English) GPT built from scratch in PyTorch: tokenizer, model,
pretraining, LoRA/SFT and evaluation, engineered as a tested, reproducible package.

> Status: Phase 0 (project setup). See [docs/PROBLEM.md](docs/PROBLEM.md) for the problem,
> the solution and how it is evaluated.

## Quick start

```bash
make install   # uv sync + pre-commit hooks
make env       # confirm the GPU is visible
make check     # ruff + mypy + pytest
```

## Layout

| Path | Purpose |
|------|---------|
| `src/scratchllm/` | The package (tokenizer, data, model, training, finetune, eval) |
| `configs/` | One YAML per run |
| `tests/` | Unit tests that define each component's contract |
| `docs/` | Problem statement, decisions, data plan |
| `scripts/` | One-off utilities (not part of the package) |

## License

MIT
