"""Compares generation with and without the key/value cache on real checkpoints (D-030).

Part 1: the logits. A text fed in two pieces through the cache against the same text fed whole,
in float32 (should agree to rounding, ~1e-5) and under bfloat16 autocast (both differ from float32
by ~0.2; the question is whether the cached one is further off).
Part 2: the text. The six prompts of D-022, 400 new tokens each, seed 1, with and without the cache,
greedy and sampled (temperature 0.8, top-k 40), float32 and bfloat16 autocast: how many of the six
completions are identical, and how long generation took (one untimed warm-up first; GPU timings
are noisy, treat them as indications).

Part 3: time per new token, 300 tokens from an 8-token prompt, on the CPU and on the GPU.

Needs a GPU and runs/local-17m/best.pt, runs/modern-17m/best.pt.
Run from the repository root: uv run python scripts/check_kv_cache.py
"""

import time
from pathlib import Path

import torch

from scratchllm.model.generate import generate, generate_text
from scratchllm.tokenizer.tokenizer import Tokenizer
from scratchllm.training.checkpoint import load_checkpoint, model_from_state

ROOT = Path(__file__).resolve().parent.parent
DEV = torch.device("cuda")
CHECKPOINTS = [("baseline", "runs/local-17m/best.pt"), ("modern", "runs/modern-17m/best.pt")]
PROMPTS = [
    "أعلنت وزارة الصحة اليوم عن",
    "تعد مدينة القاهرة من أكبر المدن في العالم العربي",
    "النهارده أنا رايح الشغل وفي الطريق",
    "مش عارف أعمل إيه في الموضوع ده",
    "The history of the city began when",
    "In my opinion, the best way to learn",
]


def bf16():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def fp32():
    return torch.autocast("cuda", enabled=False)


def logits_part(name: str, model) -> None:
    ids = torch.randint(0, 16384, (4, 200), generator=torch.Generator().manual_seed(3)).to(DEV)
    with torch.no_grad():
        ref = model(ids)[0].float()  # float32, no cache
        with bf16():
            plain = model(ids)[0].float()
            cache = model.new_cache()
            parts = [model(ids[:, :100], cache=cache)[0].float()]
            parts += [model(ids[:, i : i + 1], cache=cache)[0].float() for i in range(100, 200)]
            cached16 = torch.cat(parts, 1)
        cache = model.new_cache()
        parts = [model(ids[:, :100], cache=cache)[0]]
        parts += [model(ids[:, i : i + 1], cache=cache)[0] for i in range(100, 200)]
        cached32 = torch.cat(parts, 1)
    print(
        f"{name:9s} logits, max |difference|: float32 cached vs uncached "
        f"{(cached32 - ref).abs().max():.1e}"
        f" | bf16 uncached vs float32 {(plain - ref).abs().max():.3f}"
        f" | bf16 cached vs float32 {(cached16 - ref).abs().max():.3f}"
    )


def text_part(name: str, model, tok: Tokenizer) -> None:
    for mode, ctx in (("float32", fp32), ("bf16", bf16)):
        with torch.no_grad(), ctx():  # warm-up, not timed
            generate_text(model, tok, PROMPTS[0], 50, temperature=0, device=DEV, use_cache=True)
            generate_text(model, tok, PROMPTS[0], 50, temperature=0, device=DEV, use_cache=False)
        for kind, temp, top_k in (("greedy", 0.0, None), ("sampled", 0.8, 40)):
            seconds = {True: 0.0, False: 0.0}
            texts: dict[bool, list[str]] = {True: [], False: []}
            for prompt in PROMPTS:
                for use in (False, True):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.no_grad(), ctx():
                        sample = generate_text(
                            model, tok, prompt, 400, temperature=temp, top_k=top_k, seed=1,
                            device=DEV, use_cache=use,
                        )  # fmt: skip
                    torch.cuda.synchronize()
                    seconds[use] += time.perf_counter() - start
                    texts[use].append(sample.completion)
            same = sum(a == b for a, b in zip(texts[False], texts[True], strict=True))
            print(
                f"{name:9s} {mode:8s} {kind:8s} identical {same}/{len(PROMPTS)}  "
                f"no cache {seconds[False]:5.1f} s  cache {seconds[True]:5.1f} s  "
                f"({seconds[False] / seconds[True]:.2f}x)"
            )


def speed_part() -> None:
    for device in (torch.device("cpu"), DEV):
        model = model_from_state(
            load_checkpoint(ROOT / "runs/modern-17m/best.pt", map_location=device), device
        )
        ids = torch.randint(0, 16384, (1, 8), generator=torch.Generator().manual_seed(1)).to(device)
        for use in (False, True):
            with torch.no_grad():
                generate(model, ids, 10, temperature=0, use_cache=use)  # warm-up
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                generate(model, ids, 300, temperature=0, use_cache=use)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            ms = 1000 * (time.perf_counter() - start) / 300
            label = "cache" if use else "no cache"
            print(f"modern    {device.type:5s} {label:8s} {ms:5.1f} ms per token")


def main() -> None:
    tok = Tokenizer.load(ROOT / "artifacts/tokenizer/bpe-16k.json")
    for name, path in CHECKPOINTS:
        model = model_from_state(load_checkpoint(ROOT / path, map_location=DEV), DEV)
        logits_part(name, model)
        text_part(name, model, tok)
    speed_part()


if __name__ == "__main__":
    main()
