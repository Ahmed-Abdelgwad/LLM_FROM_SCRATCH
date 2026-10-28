"""EDA checks that do not teach anything new about Arabic text: length and personal data."""

from __future__ import annotations

from typing import Any

from scratchllm.data.pii import find_matches


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
        self.placeholders = 0

    def update(self, doc: dict[str, Any]) -> None:
        matches = find_matches(doc["text"])
        real = [m for m in matches if not m.placeholder]
        emails = sum(m.kind == "email" for m in real)
        phones = len(real) - emails
        self.docs += 1
        self.placeholders += len(matches) - len(real)
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
            # emails at example.com/.org/.net were put there by the data source's own
            # anonymization; they are not personal data and are in none of the figures here
            "email_placeholder_matches": self.placeholders,
            # matches can pile up in a few long spam pages that repeat one number, so the rate of
            # documents with any hit is the figure to judge a source by
            "docs_with_pii": self.with_any,
            "docs_with_pii_per_1000": round(self.with_any * per_1000, 2),
            "hits_per_1000_docs": round((self.emails + self.phones) * per_1000, 2),
        }
