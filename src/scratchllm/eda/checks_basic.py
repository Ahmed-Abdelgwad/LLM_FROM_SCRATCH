"""EDA checks that do not teach anything new about Arabic text: length and personal data."""

from __future__ import annotations

import re
from typing import Any

# Arabic-Indic (U+0660-0669) and Extended Arabic-Indic (U+06F0-06F9) digits -> ASCII
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_NON_ASCII_DIGIT = re.compile("[٠-٩۰-۹]")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A phone-like number has one of these shapes. The first draft accepted ANY 8-15 digits with
# optional separators, and on real samples ~90% of its hits were year ranges ("2002-2003"), year
# pairs ("2018 2019"), dates, image ids and area codes, not phones.
_PHONE = re.compile(
    # 9-15 digits led by +, 00 or 0 (01102106655); not after a Latin letter, '.', '-' or '_'
    # (decimals, ISBNs, catalogue ids), but allowed right after an Arabic letter, where spam
    # glues numbers to words
    r"(?<![\dA-Za-z_.-])(?:\+|00|0)\d{8,14}(?![\d_])"
    r"|\+\d{1,3}(?:[\s.-]\d{2,4}){2,4}(?!\d)"  # international with groups: +20 112 074 0067
    r"|(?<!\d)0\d{1,2}[\s.-]\d{3,4}[\s.-]\d{3,4}(?!\d)"  # local with groups: 010-1234-5678
    r"|(?<!\d)\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"  # North American: (555) 123-4567
)
# dates such as 2019-05-12 or 12/05/2019 are removed first so they cannot be read as phones
_DATE = re.compile(r"(?<!\d)(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})(?!\d)")


def _quantile(sorted_values: list[int], q: float) -> int:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


class LengthStats:
    """Document length (acceptance criterion A5: median length after cleaning >= 300 chars)."""

    name = "length"

    def __init__(self, short_chars: int = 300) -> None:
        self.short_chars = short_chars
        self.chars: list[int] = []
        self.words = 0

    def update(self, doc: dict[str, Any]) -> None:
        text = doc["text"]
        self.chars.append(len(text))
        self.words += len(text.split())

    def result(self) -> dict[str, Any]:
        if not self.chars:
            return {"docs": 0}
        c = sorted(self.chars)
        return {
            "docs": len(c),
            "chars_total": sum(c),
            "words_total": self.words,
            "chars_p10": _quantile(c, 0.10),
            "chars_median": _quantile(c, 0.50),
            "chars_p90": _quantile(c, 0.90),
            "chars_max": c[-1],
            "share_shorter_than": {
                str(self.short_chars): sum(x < self.short_chars for x in c) / len(c)
            },
        }


class PersonalData:
    """Emails and phone-like numbers (acceptance criterion A7: <= 5 hits per 1,000 docs).

    Regex heuristics, so they over- and under-count: a long number such as an ID or a price list
    can look like a phone number, and unusual formats are missed. Treat the rates as a
    screening signal, not a precise count of personal data.
    """

    name = "personal_data"

    def __init__(self) -> None:
        self.docs = 0
        self.with_email = 0
        self.with_phone = 0
        self.with_any = 0
        self.emails = 0
        self.phones = 0

    def update(self, doc: dict[str, Any]) -> None:
        text = doc["text"]
        if _NON_ASCII_DIGIT.search(text):  # translating every document costs ~6 s per 100M chars
            text = text.translate(_DIGITS)
        emails = len(_EMAIL.findall(text)) if "@" in text else 0  # no '@', no email
        phones = len(_PHONE.findall(_DATE.sub(" ", text)))
        self.docs += 1
        self.with_email += emails > 0
        self.with_phone += phones > 0
        self.with_any += (emails + phones) > 0
        self.emails += emails
        self.phones += phones

    def result(self) -> dict[str, Any]:
        per_1000 = 1000 / self.docs if self.docs else 0.0
        return {
            "docs": self.docs,
            "docs_with_email": self.with_email,
            "docs_with_phone_like": self.with_phone,
            "email_matches": self.emails,
            "phone_like_matches": self.phones,
            # matches can pile up in a few long spam pages that repeat one number, so the rate of
            # documents with any hit is the figure to judge a source by
            "docs_with_pii": self.with_any,
            "docs_with_pii_per_1000": round(self.with_any * per_1000, 2),
            "hits_per_1000_docs": round((self.emails + self.phones) * per_1000, 2),
        }
