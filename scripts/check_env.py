"""Print the facts that decide our compute budget. Run: uv run python scripts/check_env.py"""

import platform

import torch


def main() -> None:
    print(f"python  : {platform.python_version()}")
    print(f"torch   : {torch.__version__}")
    print(f"cuda    : {torch.cuda.is_available()} (build {torch.version.cuda})")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        mem_gib = p.total_memory / 2**30
        print(f"gpu     : {p.name}, {mem_gib:.1f} GiB, capability {p.major}.{p.minor}")
        print(f"bf16    : {torch.cuda.is_bf16_supported()}")
    else:
        print("WARNING: no CUDA device visible; training would fall back to CPU")


if __name__ == "__main__":
    main()
