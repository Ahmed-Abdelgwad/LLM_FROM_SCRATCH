"""Finding emails and phone-like numbers in text, with their positions.

One definition serves both the EDA check (how common is personal data, A7) and the cleaning
rule (mask it), so what is measured is what is removed. These are regular expressions, so they
are a screening tool and not a guarantee: formats nobody thought of are missed, and a long ID
can look like a phone number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Arabic-Indic (U+0660-0669) and Extended Arabic-Indic (U+06F0-06F9) digits -> ASCII. The map is
# one character to one character, so a position in the translated text is the same position in
# the original.
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
NON_ASCII_DIGIT = re.compile("[٠-٩۰-۹]")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A phone-like number has one of these shapes. The first draft accepted ANY 8-15 digits with
# optional separators, and on real samples ~90% of its hits were year ranges ("2002-2003"), year
# pairs ("2018 2019"), dates, image ids and area codes, not phones.
PHONE = re.compile(
    # 9-15 digits led by +, 00 or 0 (01102106655); not after a Latin letter, '.', '-' or '_'
    # (decimals, ISBNs, catalogue ids), but allowed right after an Arabic letter, where spam
    # glues numbers to words
    r"(?<![\dA-Za-z_.-])(?:\+|00|0)\d{8,14}(?![\d_])"
    r"|\+\d{1,3}(?:[\s.-]\d{2,4}){2,4}(?!\d)"  # international with groups: +20 112 074 0067
    r"|(?<!\d)0\d{1,2}[\s.-]\d{3,4}[\s.-]\d{3,4}(?!\d)"  # local with groups: 010-1234-5678
    r"|(?<!\d)\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"  # North American: (555) 123-4567
)
# dates such as 2019-05-12 or 12/05/2019 are blanked first so they cannot be read as phones
DATE = re.compile(r"(?<!\d)(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})(?!\d)")

# The domains example.com, .org and .net are reserved (RFC 2606) and cannot be a real person's
# address. The FineWeb pipelines already replaced real emails by `email@example.com` and
# `firstname.lastname@example.org`, so 89% to 100% of the emails we find are these placeholders.
EXAMPLE_DOMAIN = re.compile(r"@example\.(?:com|org|net)$", re.IGNORECASE)

EMAIL_PLACEHOLDER = "<email>"
PHONE_PLACEHOLDER = "<phone>"


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    kind: str  # "email" or "phone"
    placeholder: bool = False  # an email at a reserved example domain: not personal data


def find_matches(text: str) -> list[Match]:
    """Every email and phone-like number in `text`, ordered by position.

    An email's digits can also look like a phone number, so the two kinds may overlap; both are
    returned (the EDA check counts both) and `mask` merges them. A placeholder email is returned
    too, flagged, so that it can be masked like the others but not counted as personal data.
    """
    t = text.translate(DIGITS) if NON_ASCII_DIGIT.search(text) else text  # same length as text
    found: list[Match] = []
    if "@" in t:  # no '@', no email
        found += [
            Match(m.start(), m.end(), "email", bool(EXAMPLE_DOMAIN.search(m.group())))
            for m in EMAIL.finditer(t)
        ]
    blanked = DATE.sub(lambda m: " " * (m.end() - m.start()), t)  # same length: positions hold
    found += [Match(m.start(), m.end(), "phone") for m in PHONE.finditer(blanked)]
    return sorted(found, key=lambda m: (m.start, m.end))


def mask(text: str, matches: list[Match]) -> str:
    """Replace each match by its placeholder. Overlapping matches become one; email wins."""
    merged: list[Match] = []
    for m in matches:  # already ordered by start
        if merged and m.start < merged[-1].end:
            last = merged[-1]
            kind = "email" if "email" in (last.kind, m.kind) else "phone"
            merged[-1] = Match(
                last.start, max(last.end, m.end), kind, last.placeholder and m.placeholder
            )
        else:
            merged.append(m)
    pieces: list[str] = []
    cursor = 0
    for m in merged:
        pieces.append(text[cursor : m.start])
        pieces.append(EMAIL_PLACEHOLDER if m.kind == "email" else PHONE_PLACEHOLDER)
        cursor = m.end
    pieces.append(text[cursor:])
    return "".join(pieces)
