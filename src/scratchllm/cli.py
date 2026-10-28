"""Command-line entry point: `scratchllm --help`."""

from __future__ import annotations

import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from scratchllm.data.leakage import collect_eval_hashes, count_train_leaks, purge_train_leaks
from scratchllm.data.sampling import (
    SampleSpec,
    build_docs,
    iter_source_rows,
    latest_revision,
    open_hf,
    verify_manifest,
    write_sample,
)
from scratchllm.data.sources import SOURCES, get_source

app = typer.Typer(no_args_is_help=True, help="scratchllm: bilingual GPT from scratch.")
data_app = typer.Typer(no_args_is_help=True, help="Data acquisition and analysis.")
app.add_typer(data_app, name="data")


@data_app.command("sources")
def list_sources() -> None:
    """List the registered corpora."""
    for s in SOURCES.values():
        typer.echo(f"{s.name:18} {s.repo}/{s.config:14} {s.language} {s.variety:16} {s.license}")


@data_app.command("sample")
def sample(
    source: Annotated[str, typer.Argument(help="Registered source name (see `data sources`).")],
    n_docs: Annotated[int, typer.Option(help="Documents to draw.")] = 50_000,
    seed: int = 0,
    val_frac: float = 0.02,
    test_frac: float = 0.02,
    out_dir: Path = Path("data/raw/eda"),
    rows_per_group: Annotated[
        int, typer.Option(help="Rows taken from each sampled row group.")
    ] = 250,
    workers: Annotated[int, typer.Option(help="Parallel row-group reads.")] = 4,
    cache_dir: Annotated[
        Path, typer.Option(help="Where parquet footers are cached between draws.")
    ] = Path("data/cache/footers"),
    revision: Annotated[
        str | None, typer.Option(help="Pin a dataset commit; default is latest.")
    ] = None,
) -> None:
    """Draw a seeded sample spread over the whole source, label splits, write with a manifest."""
    code = 0
    try:
        src = get_source(source)
        spec = SampleSpec(
            n_docs=n_docs,
            seed=seed,
            val_frac=val_frac,
            test_frac=test_frac,
            rows_per_group=rows_per_group,
            read_workers=workers,
        )
        rev = revision or latest_revision(src.repo)
        typer.echo(f"{src.name}: {src.repo} @ {rev[:10]}, drawing {n_docs} docs (seed {seed})")
        stats: Counter[str] = Counter()
        fs, root = open_hf(src, rev)
        rows = iter_source_rows(src, spec, stats, fs, root, cache_dir)
        docs = build_docs(rows, src, spec, stats)
        m = write_sample(docs, out_dir, src, spec, rev, stats)
        typer.echo(f"wrote {m.n_docs} docs {m.docs_per_split}; sha256 {m.sha256[:12]}")
        typer.echo(f"stats: {m.stats}")
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        _exit_now(code)


def _exit_now(code: int) -> NoReturn:
    """End the process without interpreter finalization.

    We stop reading early, which can leave remote reads in flight. With the old HF streaming
    reader this deadlocked at exit: an Arrow I/O worker was running Python code (a download
    retry sleeping) while the main thread sat in arrow::internal::ThreadPool::Shutdown waiting
    for it, and the worker waited for the finalizing interpreter's GIL (confirmed with gdb).
    The row-group reader is a different code path and has not been shown to hang, but it also
    abandons in-flight reads, so we keep the hard exit. Output is already fsynced and atomically
    renamed by write_sample, so exiting here is safe.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


@data_app.command("analyze")
def analyze(
    manifests: list[Path],
    out_dir: Annotated[Path, typer.Option(help="Where reports are written.")] = Path("reports/eda"),
    run_id: Annotated[
        str | None,
        typer.Option(
            help="Report folder name (default: UTC timestamp). Use final-<name> for a "
            "report that should be tracked in git."
        ),
    ] = None,
) -> None:
    """Run the EDA checks on the train split of each sample and write report.json / report.md."""
    from scratchllm.eda.runner import run_eda

    target = run_eda(manifests, out_dir, run_id=run_id, progress=typer.echo)
    typer.echo(f"report written to {target}")


@data_app.command("read")
def read(
    manifest: Path,
    kind: Annotated[str, typer.Option(help="all, mixed (Arabic and Latin) or short.")] = "all",
    n: Annotated[int, typer.Option(help="Documents to draw.")] = 50,
    seed: int = 0,
    chars: Annotated[int, typer.Option(help="Characters shown per document.")] = 300,
    out: Annotated[Path | None, typer.Option(help="Write here instead of the screen.")] = None,
) -> None:
    """Draw train documents at random to read by hand (same arguments, same documents)."""
    from scratchllm.eda.reading import KINDS, read_sample, render

    if kind not in KINDS:
        raise typer.BadParameter(f"kind must be one of {', '.join(KINDS)}")
    docs, total = read_sample(manifest, kind, n, seed)
    text = render(docs, total, chars, f"{manifest.name}: {kind}, seed {seed}")
    if out is None:
        typer.echo(text)
    else:
        out.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}")


@data_app.command("clean-report")
def clean_report(manifests: list[Path]) -> None:
    """Show what the cleaning rules would do to the train split of each sample (writes nothing)."""
    from scratchllm.eda.cleaning_effect import measure

    for path in manifests:
        r = measure(path)
        rules = ", ".join(r["rules"]) or "none measured yet"
        typer.echo(f"{r['source']}: rules: {rules}")
        typer.echo(
            f"  documents {r['docs_in']} -> {r['docs_out']} (retention {r['retention']:.1%}); "
            f"characters {r['chars_in']:,} -> {r['chars_out']:,}"
        )
        typer.echo(
            f"  median length {r['median_chars_before']} -> {r['median_chars_after']}; "
            f"under 300 characters {r['share_under_300_before']:.1%} -> "
            f"{r['share_under_300_after']:.1%}"
        )
        for name in r["rules"]:
            typer.echo(
                f"  {name}: dropped {r['dropped'].get(name, 0)} documents, shortened "
                f"{r['changed'].get(name, 0)}, removed "
                f"{r['chars_removed'].get(name, 0):,} characters"
            )


@data_app.command("verify")
def verify(manifests: list[Path]) -> None:
    """Check data files still match their manifests (hash and counts)."""
    for path in manifests:
        m = verify_manifest(path)
        typer.echo(f"OK {path.name}: {m.n_docs} docs")


@data_app.command("leaks")
def leaks(
    manifests: list[Path],
    purge_to: Annotated[Path | None, typer.Option(help="Write leak-free copies here.")] = None,
) -> None:
    """Report (and optionally remove) train docs that duplicate an eval doc, across sources."""
    eval_hashes = collect_eval_hashes(manifests)
    typer.echo(f"{len(eval_hashes)} distinct eval texts")
    typer.echo(f"train leaks per source: {count_train_leaks(manifests, eval_hashes)}")
    if purge_to is not None:
        for name, r in purge_train_leaks(manifests, eval_hashes, purge_to).items():
            typer.echo(f"{name}: kept {r['kept']}, removed {r['removed_train_leaks']}")
