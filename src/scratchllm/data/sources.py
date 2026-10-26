"""Registry of corpora we may draw from. Mirrors docs/data/SOURCES.md; keep them in sync."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    repo: str  # Hugging Face dataset id
    config: str
    language: str  # "ar" | "en"
    variety: str  # "msa" | "egyptian" | ... | "mixed" | "en"
    license: str
    kind: str  # "web" | "wiki"
    split: str = "train"
    text_field: str = "text"
    id_field: str = "id"
    url_field: str | None = "url"
    data_glob: str = ""  # parquet files inside the repo, e.g. "data/arz_Arab/train/*.parquet"


_FW2 = "HuggingFaceFW/fineweb-2"
_WIKI = "wikimedia/wikipedia"


def _fw2(config: str, variety: str) -> Source:
    return Source(
        f"fw2_{config[:3]}",
        _FW2,
        config,
        "ar",
        variety,
        "ODC-By",
        "web",
        data_glob=f"data/{config}/train/*.parquet",
    )


def _wiki(code: str, language: str, variety: str) -> Source:
    return Source(
        f"wiki_{code}",
        _WIKI,
        f"20231101.{code}",
        language,
        variety,
        "CC-BY-SA-3.0 + GFDL",
        "wiki",
        data_glob=f"20231101.{code}/*.parquet",
    )


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        _fw2("arb_Arab", "msa"),
        _fw2("arz_Arab", "egyptian"),
        _fw2("ary_Arab", "moroccan"),
        _fw2("ars_Arab", "najdi"),
        _fw2("apc_Arab", "north_levantine"),
        _fw2("acm_Arab", "mesopotamian"),
        _wiki("ar", "ar", "msa"),
        _wiki("arz", "ar", "egyptian"),
        _wiki("ary", "ar", "moroccan"),
        _wiki("en", "en", "en"),
        Source(
            "fineweb_edu_10bt",
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            "en",
            "en",
            "ODC-By",
            "web",
            data_glob="sample/10BT/*.parquet",
        ),
    ]
}


def get_source(name: str) -> Source:
    try:
        return SOURCES[name]
    except KeyError:
        raise KeyError(f"unknown source {name!r}; known: {', '.join(sorted(SOURCES))}") from None
