"""Text normalization applied before the tokenizer is trained and before the model reads text.

Everything here is chosen because it changes the text in a way nobody reads, and nothing that
changes a word (decision D-006): the alef, yeh and teh-marbuta spellings, the diacritics and the
digits are left exactly as written.

The steps run in this order, and the order matters:

1. Arabic presentation forms -> ordinary letters (NFKC, applied to those characters only).
2. Remove the tatweel.
3. Whitespace.

Why presentation forms, and why only them. Text copied from PDFs and old web pages carries
isolated, initial, medial and final glyph forms (U+FB50-FDFF, U+FE70-FEFF) instead of the
letters, so the same word has several spellings: the ligature U+FEFB is two letters "لا", and
"ﷲ" is a single code point for "الله". NFKC maps them back. They are rare in modern standard
Arabic (0.04% of the characters of fw2_arb) but not in popular web text (6% of the documents of
fw2_apc). NFKC is NOT applied to the rest of the text, because on this corpus it would also turn
"…" into "...", "m²" into "m2" and "™" into "TM" in the English text, which loses information
for nothing.

Why the order. NFKC of some presentation forms contains a tatweel: U+FE77 (fatha medial form)
becomes tatweel + fatha. If the tatweel were removed first, those would survive.
"""

from __future__ import annotations

import re
import unicodedata

# Arabic Presentation Forms-A (U+FB50-FDFF) and -B (U+FE70-FEFF).
_PRESENTATION_FORMS = re.compile("[ﭐ-﷿ﹰ-﻿]")
_TATWEEL = "ـ"
# Any whitespace except the newline: space, tab, no-break space, carriage return, form feed...
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_SPACE_AROUND_NEWLINE = re.compile(r" ?\n ?")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def _nfkc(match: re.Match[str]) -> str:
    return unicodedata.normalize("NFKC", match.group())


def normalize(text: str) -> str:
    """Return `text` with presentation forms resolved, no tatweel, and tidy whitespace.

    Idempotent: normalize(normalize(x)) == normalize(x).
    """
    text = _PRESENTATION_FORMS.sub(_nfkc, text)
    text = text.replace(_TATWEEL, "")
    text = _HORIZONTAL_SPACE.sub(" ", text)  # runs of spaces and tabs, and no-break spaces
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)  # no space left at the start or end of a line
    text = _MANY_NEWLINES.sub("\n\n", text)  # a paragraph break is two newlines, never more
    return text.strip()
