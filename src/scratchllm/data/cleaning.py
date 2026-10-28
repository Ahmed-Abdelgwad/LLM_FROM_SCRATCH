"""Cleaning rules: each removes something the EDA showed is not text worth learning from.

A rule is a function from a document's text to new text, or to None when the whole document
should be dropped. Rules are small and independent so that each one can be measured alone
(documents touched, characters removed) before it is accepted; that is how acceptance
criterion A1 (retention) gets its numbers. Only rules backed by a measurement on the samples
are here, and a source with no measured rule is left untouched rather than guessed at.

Every rule says why it exists; the evidence is in docs/DECISIONS.md (D-004, D-005).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from scratchllm.data.sources import Source


@dataclass(frozen=True)
class Rule:
    name: str
    run: Callable[[str], str | None]  # new text; the same text if nothing applies; None = drop


# ---------------------------------------------------------------------------------------------
# Rule 1: Google Books search stubs (D-004)
#
# Purpose: a Google Books "search inside this book" page has no text of its own. Its body is a
# fixed sentence ("What people are saying - Write a review ...") followed by a list of two-letter
# Arabic fragments (the book's word cloud). Training on it teaches the model that such strings
# are Arabic, and it makes the dialect corpus look mixed-script and low quality.
# Benefit: removes 382 of 9,670 fw2_apc train documents (3.95%), 28 of fineweb_edu_10bt's
# 47,939, and none elsewhere, so it costs almost nothing and removes a clear contaminant.
#
# Decision for how to recognise it: by the stub sentences, not by the web address. The same
# domain also holds ejabat.google.com question-and-answer threads, which are real Levantine text.
# One sentence can appear in a page that quotes it, so two of the three are required.
# ---------------------------------------------------------------------------------------------
BOOKS_STUB_PHRASES = (
    "What people are saying - Write a review",
    "Try this search over all volumes",
    "We haven't found any reviews in the usual places",
)
BOOKS_STUB_MIN_PHRASES = 2


def is_books_stub(text: str) -> bool:
    return sum(phrase in text for phrase in BOOKS_STUB_PHRASES) >= BOOKS_STUB_MIN_PHRASES


def drop_books_stub(text: str) -> str | None:
    return None if is_books_stub(text) else text


# ---------------------------------------------------------------------------------------------
# Rule 2: the footer of a Wikipedia article (D-005)
#
# Purpose: a Wikipedia article in this dump ends with headings such as "references" and "external
# links" followed only by the names of its categories ("Villages of Sétif Province", "1951
# births"), which are not sentences. Everything from the first such heading on is dropped.
# Benefits: (1) the model stops learning category lists as if they were prose; (2) document
# length becomes the length of the prose, so the minimum-length question (A5) is asked of the
# right quantity: a 400-character page may be 150 characters of text and 250 of categories;
# (3) templated stubs become more alike, which makes the near-duplicate purge find them.
# Measured effect on the samples: about 16% of wiki_ar's characters and 9% of wiki_ary's.
#
# The heading lists come from counting the most frequent short lines of each source (66.5% of
# wiki_ar documents have a line "مراجع", 96.6% of wiki_ary documents "عيون لكلام"). A source
# whose headings have not been measured gets no footer rule.
#
# Conservative on purpose: if the part after the heading still has a very long line, it is
# probably text and not a list (a section placed after the references, or an unusually long
# citation), and the document is left unchanged. If nothing precedes the first heading the
# document has no prose and is dropped.
# ---------------------------------------------------------------------------------------------
WIKI_FOOTER_HEADINGS: dict[str, frozenset[str]] = {
    "wiki_ar": frozenset(
        {
            "مراجع",
            "المراجع",
            "مصادر",
            "المصادر",
            "وصلات خارجية",
            "روابط خارجية",
            "مراجع وروابط خارجية",
            "انظر أيضًا",
            "انظر أيضاً",
            "انظر أيضا",
            "طالع أيضا",
        }
    ),
    "wiki_ary": frozenset({"عيون لكلام", "نوطات", "شوف تا"}),
}
FOOTER_MAX_LINE_CHARS = 300


def strip_wiki_footer(
    text: str, headings: frozenset[str], max_line_chars: int = FOOTER_MAX_LINE_CHARS
) -> str | None:
    lines = text.split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip() in headings), None)
    if first is None:
        return text
    if any(len(line) > max_line_chars for line in lines[first + 1 :]):
        return text  # not a plain list of names: leave it
    prose = "\n".join(lines[:first]).rstrip()
    return prose or None


def rules_for(source: Source) -> list[Rule]:
    """The measured rules that apply to `source`; an empty list when none has been measured."""
    rules: list[Rule] = []
    if source.kind == "web":
        rules.append(Rule("books_stub", drop_books_stub))
    headings = WIKI_FOOTER_HEADINGS.get(source.name)
    if source.kind == "wiki" and headings is not None:
        rules.append(Rule("wiki_footer", partial(strip_wiki_footer, headings=headings)))
    return rules


@dataclass
class CleaningStats:
    """What the rules did, so retention (A1) is a measured number and not a guess."""

    docs_in: int = 0
    docs_out: int = 0
    chars_in: int = 0
    chars_out: int = 0
    dropped: Counter[str] = field(default_factory=Counter)  # rule -> documents dropped
    changed: Counter[str] = field(default_factory=Counter)  # rule -> documents shortened
    chars_removed: Counter[str] = field(default_factory=Counter)  # rule -> characters removed


def clean_docs(
    docs: Iterable[dict[str, Any]], rules: list[Rule], stats: CleaningStats
) -> Iterator[dict[str, Any]]:
    """Apply the rules in order. Yields new records; `text_sha256` still names the raw text."""
    for doc in docs:
        text = doc["text"]
        stats.docs_in += 1
        stats.chars_in += len(text)
        dropped = False
        for rule in rules:
            result = rule.run(text)
            if result is None:
                stats.dropped[rule.name] += 1
                stats.chars_removed[rule.name] += len(text)
                dropped = True
                break
            if len(result) < len(text):
                stats.changed[rule.name] += 1
                stats.chars_removed[rule.name] += len(text) - len(result)
                text = result
        if dropped:
            continue
        stats.docs_out += 1
        stats.chars_out += len(text)
        yield {**doc, "text": text}
