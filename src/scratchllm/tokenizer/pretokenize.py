"""Pre-tokenization: cutting text into the pieces inside which BPE is allowed to merge.

BPE never merges across two pieces, so this step decides what a "word" is before any statistics
are counted. The pattern is GPT-2's (Radford et al. 2019, section 2.2) with one change that
Arabic needs (decision D-014): combining marks (\\p{M}) count as part of a word.

GPT-2 writes a word as \\p{L}+, letters only. Arabic diacritics (fatha, damma, shadda, the
dagger alef...) are combining marks, not letters, so "مُحَمَّد" would be cut at every diacritic
into pieces that start with a bare mark. Measured on 1,500 documents per source: that adds 9.2%
more pieces on fw2_arb, 3.8% on wiki_ar, 2.8% on fw2_arz, and 0.02% on English (fineweb_edu),
where the change is nearly invisible. With \\p{M} the Arabic word stays one piece.

What the pieces are, in order of the alternatives: an English contraction ("'s", "'re"), a run of
letters/marks, a run of digits, a run of other symbols, a run of whitespace that is not followed
by a word, a run of whitespace. A single space in front of a word or number belongs to it, which
is what lets " the" and " مرحبا" be single tokens.
"""

from __future__ import annotations

import regex

PATTERN = regex.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[\p{L}\p{M}]+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{M}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def pretokenize(text: str) -> list[str]:
    """Cut `text` into pieces. Joining the pieces gives `text` back, character for character."""
    return PATTERN.findall(text)
