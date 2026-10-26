"""The three EDA checks about Arabic text.

Each is a contract: `result()` returns exactly the keys listed in the class docstring (the
report and the later decisions read them). Until a check is implemented its `update` raises
NotImplementedError, and the runner reports it as "not_implemented" instead of failing.

Status: ScriptMix and SurfaceVariation are implemented (written by the assistant at the owner's
request, with the decisions explained in their docstrings). NearDuplicates is still a contract:
write its tests first, with hand-made strings whose answers you know (see
tests/test_eda_learner.py for the style), then make them pass.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

# Unicode blocks whose letters belong to the Arabic script: the main block, its supplement,
# extended-A, and the two presentation-form blocks (what copy-pasted PDF text often contains,
# e.g. the ligature U+FEFB).
_ARABIC_BLOCKS = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_TATWEEL = "ـ"


def letter_script(ch: str) -> str | None:
    """'arabic', 'latin' or 'other' for a letter; None for anything that is not a letter.

    Decisions, all deliberate:
    - "Letter" means str.isalpha(). Digits, spaces, punctuation and the combining marks
      (harakat, shadda) are not letters, so a vowelled word has the same letter count as an
      unvowelled one.
    - The tatweel (U+0640) is NOT counted. isalpha() says it is a letter, but it is a spacing
      stretch with no letter identity, and counting it would let "مـــــرحبا" outweigh "مرحبا".
    - The script is decided by Unicode block, so Persian and Urdu letters (پ چ گ ک) count as
      Arabic *script*. This measures script, not language.
    - A non-ASCII letter is Latin if its Unicode name starts with "LATIN" (é, ñ, ß ...).
    """
    if not ch.isalpha() or ch == _TATWEEL:
        return None
    code = ord(ch)
    if any(lo <= code <= hi for lo, hi in _ARABIC_BLOCKS):
        return "arabic"
    if code < 0x80 or unicodedata.name(ch, "").startswith("LATIN"):
        return "latin"
    return "other"


class ScriptMix:
    """EDA question #1 / acceptance criteria A3 and A4: which scripts do the letters use?

    For each document, count its *letters* (see `letter_script` for exactly what that means) and
    split them by script into Arabic, Latin and other. A document's share is letters-in-that-
    script divided by all its letters. A document with no letters at all is counted separately and
    is left out of every share and count below, because its shares are undefined.

    result() keys:
      docs                         number of documents seen
      docs_without_letters         documents with no letters
      docs_with_letters            docs - docs_without_letters (the denominator of the means)
      mean_arabic_share            average of the per-document Arabic share
      mean_latin_share             average of the per-document Latin share
      mean_other_share             average of the per-document share of every other script
      letters_total                pooled letter counts {"arabic", "latin", "other"}; long documents
                                   dominate these, which is why the means above are per document
      docs_arabic_share_ge_0_8     count with Arabic share >= 0.8       (A3, Arabic sources)
      docs_latin_share_ge_0_9      count with Latin share  >= 0.9       (A4, English sources)
      docs_mixed                   count with Arabic share >= 0.2 AND Latin share >= 0.2

    The threshold comparisons use integers (e.g. arabic * 5 >= letters * 4 for 0.8), so a
    document at exactly 80% is counted and float rounding can never move it across the line.
    """

    name = "script_mix"

    def __init__(self) -> None:
        self.docs = 0
        self.docs_without_letters = 0
        self.letters_total: Counter[str] = Counter()
        self._share_sums = {"arabic": 0.0, "latin": 0.0, "other": 0.0}
        self._arabic_ge_08 = self._latin_ge_09 = self._mixed = 0
        self._script_of_char: dict[str, str | None] = {}  # classify each distinct character once

    def update(self, doc: dict[str, Any]) -> None:
        self.docs += 1
        per_script: Counter[str] = Counter()
        # Counter(text) counts at C speed; only the ~100 distinct characters get classified
        for ch, n in Counter(doc["text"]).items():
            if ch not in self._script_of_char:
                self._script_of_char[ch] = letter_script(ch)
            script = self._script_of_char[ch]
            if script is not None:
                per_script[script] += n
        letters = sum(per_script.values())
        if letters == 0:
            self.docs_without_letters += 1
            return
        self.letters_total.update(per_script)
        for script in self._share_sums:
            self._share_sums[script] += per_script[script] / letters
        arabic, latin = per_script["arabic"], per_script["latin"]
        self._arabic_ge_08 += arabic * 5 >= letters * 4  # arabic / letters >= 0.8
        self._latin_ge_09 += latin * 10 >= letters * 9  # latin / letters >= 0.9
        self._mixed += arabic * 5 >= letters and latin * 5 >= letters  # both >= 0.2

    def result(self) -> dict[str, Any]:
        with_letters = self.docs - self.docs_without_letters
        mean = {
            s: (total / with_letters if with_letters else 0.0)
            for s, total in self._share_sums.items()
        }
        return {
            "docs": self.docs,
            "docs_without_letters": self.docs_without_letters,
            "docs_with_letters": with_letters,
            "mean_arabic_share": round(mean["arabic"], 4),
            "mean_latin_share": round(mean["latin"], 4),
            "mean_other_share": round(mean["other"], 4),
            "letters_total": {s: self.letters_total[s] for s in ("arabic", "latin", "other")},
            "docs_arabic_share_ge_0_8": self._arabic_ge_08,
            "docs_latin_share_ge_0_9": self._latin_ge_09,
            "docs_mixed": self._mixed,
        }


# Single characters whose occurrences are counted directly. Code points are written as numbers
# so the source never contains an invisible or easily mistaken character.
_SINGLE_CHAR_ITEMS: dict[str, tuple[str, ...]] = {
    # fathatan, dammatan, kasratan, fatha, damma, kasra, sukun (not the shadda U+0651)
    "diacritics_without_shadda": tuple(
        chr(c) for c in (0x064B, 0x064C, 0x064D, 0x064E, 0x064F, 0x0650, 0x0652)
    ),
    "shadda": (chr(0x0651),),
    "tatweel": (chr(0x0640),),
    "alef_plain": (chr(0x0627),),
    "alef_hamza_above": (chr(0x0623),),
    "alef_hamza_below": (chr(0x0625),),
    "alef_madda": (chr(0x0622),),
    "yeh": (chr(0x064A),),
    "alef_maqsura": (chr(0x0649),),
    "digits_ascii": tuple("0123456789"),
    "digits_arabic_indic": tuple(chr(c) for c in range(0x0660, 0x066A)),
    "digits_extended_arabic_indic": tuple(chr(c) for c in range(0x06F0, 0x06FA)),
}
_TEH_MARBUTA, _HEH = chr(0x0629), chr(0x0647)
# Arabic combining marks (U+064B-065F), dagger alef (U+0670) and the tatweel: removed before
# looking for word-final letters, so "مدرسةٌ" and "مدرسةـ" still end in the teh marbuta.
_MARKS_AND_TATWEEL = re.compile(f"[{chr(0x064B)}-{chr(0x065F)}{chr(0x0670)}{chr(0x0640)}]")
_TEH_MARBUTA_FINAL = re.compile(_TEH_MARBUTA + r"(?!\w)")
_HEH_FINAL = re.compile(_HEH + r"(?!\w)")
_ITEM_ORDER = (
    "diacritics_without_shadda",
    "shadda",
    "tatweel",
    "alef_plain",
    "alef_hamza_above",
    "alef_hamza_below",
    "alef_madda",
    "yeh",
    "alef_maqsura",
    "teh_marbuta_final",
    "heh_final",
    "digits_ascii",
    "digits_arabic_indic",
    "digits_extended_arabic_indic",
)


class SurfaceVariation:
    """EDA question #4: how inconsistently is Arabic spelled? (decides the normalization rules)

    Counts, over the whole sample, how often each spelling variant appears, so we can decide
    which ones to unify. For every item it reports `docs_with` (documents containing it at least
    once) and `per_1000_letters` (occurrences per 1,000 Arabic letters in the whole sample;
    0.0 when the sample has no Arabic letters). An Arabic letter is what `letter_script` calls
    one, the same definition ScriptMix uses.

    result() keys:
      docs                    number of documents seen
      arabic_letters          total Arabic letters counted (the denominator of every rate)
      items                   maps each name below to {"docs_with": int, "per_1000_letters": float}
      pair_shares             how split the spelling is within a pair (None if the pair is absent)

    Items:
      diacritics_without_shadda   fathatan, dammatan, kasratan, fatha, damma, kasra, sukun
                                  (U+064B-0650 and U+0652). Other marks, e.g. the dagger alef, are
                                  not counted here.
      shadda                      reported on its own: dialect spelling uses it, so it must not be
                                  lumped with the other diacritics
      tatweel                     the elongation character
      alef_plain, alef_hamza_above, alef_hamza_below, alef_madda      the alef forms
      yeh, alef_maqsura           ي versus ى
      teh_marbuta_final, heh_final   ة versus ه as the LAST letter of a word. Marks and tatweel
                                  after the letter are ignored; a following letter or digit
                                  means the letter is inside the word. Note that ه is also the
                                  last letter of legitimate words such as "الله", so heh_final is
                                  an upper bound on how often ه stands where ة could.
      digits_ascii, digits_arabic_indic, digits_extended_arabic_indic

    pair_shares (each is the second member's share of the pair; 0.5 means evenly split):
      alef_maqsura_among_yeh_like        alef_maqsura / (yeh + alef_maqsura)
      heh_among_final_teh_heh            heh_final / (teh_marbuta_final + heh_final)
      alef_plain_among_alef_forms        alef_plain / all four alef forms
      shadda_among_all_diacritics        shadda / (shadda + diacritics_without_shadda)

    Why it matters: the same word spelled two ways becomes two different tokens. If the rates
    are high and the variants are interchangeable, normalizing them lowers tokenizer fertility;
    if a variant is meaningful (shadda in dialects), normalizing would destroy information.
    """

    name = "surface_variation"

    def __init__(self) -> None:
        self.docs = 0
        self.arabic_letters = 0
        self._occurrences = dict.fromkeys(_ITEM_ORDER, 0)
        self._docs_with = dict.fromkeys(_ITEM_ORDER, 0)
        self._is_arabic_letter: dict[str, bool] = {}  # classify each distinct character once

    def _add(self, item: str, occurrences: int) -> None:
        self._occurrences[item] += occurrences
        self._docs_with[item] += occurrences > 0

    def update(self, doc: dict[str, Any]) -> None:
        self.docs += 1
        text = doc["text"]
        counts = Counter(text)  # C-speed; the per-character questions below are dictionary lookups
        for ch, n in counts.items():
            if ch not in self._is_arabic_letter:
                self._is_arabic_letter[ch] = letter_script(ch) == "arabic"
            if self._is_arabic_letter[ch]:
                self.arabic_letters += n
        for item, chars in _SINGLE_CHAR_ITEMS.items():
            self._add(item, sum(counts.get(c, 0) for c in chars))
        teh_or_heh = counts.get(_TEH_MARBUTA, 0) or counts.get(_HEH, 0)
        stripped = _MARKS_AND_TATWEEL.sub("", text) if teh_or_heh else ""
        self._add("teh_marbuta_final", len(_TEH_MARBUTA_FINAL.findall(stripped)))
        self._add("heh_final", len(_HEH_FINAL.findall(stripped)))

    def _share(self, part: str, *whole: str) -> float | None:
        total = sum(self._occurrences[w] for w in whole)
        return round(self._occurrences[part] / total, 4) if total else None

    def result(self) -> dict[str, Any]:
        letters = self.arabic_letters
        return {
            "docs": self.docs,
            "arabic_letters": letters,
            "items": {
                item: {
                    "docs_with": self._docs_with[item],
                    "per_1000_letters": round(1000 * self._occurrences[item] / letters, 3)
                    if letters
                    else 0.0,
                }
                for item in _ITEM_ORDER
            },
            "pair_shares": {
                "alef_maqsura_among_yeh_like": self._share("alef_maqsura", "yeh", "alef_maqsura"),
                "heh_among_final_teh_heh": self._share(
                    "heh_final", "teh_marbuta_final", "heh_final"
                ),
                "alef_plain_among_alef_forms": self._share(
                    "alef_plain", "alef_plain", "alef_hamza_above", "alef_hamza_below", "alef_madda"
                ),
                "shadda_among_all_diacritics": self._share(
                    "shadda", "shadda", "diacritics_without_shadda"
                ),
            },
        }


class NearDuplicates:
    """EDA question #3 / acceptance criterion A2: how much of the sample is near-duplicate?

    Two documents are near-duplicates when the Jaccard similarity of their sets of word
    shingles is >= `threshold`. Exact duplicates were already removed at sampling time, so this
    finds templated pages (same text with a few words changed).

    Implement MinHash signatures and LSH banding yourself (no extra dependency). A signature
    summarises a document's shingle set in `num_perm` numbers; documents whose signatures agree
    in a whole band become candidate pairs; candidates are then confirmed by comparing
    signatures against `threshold`. With 50k documents everything fits in memory.

    result() keys:
      docs                          number of documents seen
      threshold, num_perm, shingle_words   the parameters actually used
      near_duplicate_docs           documents that have at least one near-duplicate neighbour
      near_duplicate_rate           near_duplicate_docs / docs        (A2 asks for <= 0.30)
      clusters                      number of groups of 2 or more near-duplicates
      largest_cluster_sizes         sizes of the 10 largest clusters, descending
      largest_cluster_examples      for each of those, the first 3 document ids and 120 chars
                                    of the first one, so a human can read what the template is
    """

    name = "near_duplicates"

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, shingle_words: int = 5) -> None:
        self.threshold, self.num_perm, self.shingle_words = threshold, num_perm, shingle_words

    def update(self, doc: dict[str, Any]) -> None:
        raise NotImplementedError("NearDuplicates.update is yours to write")

    def result(self) -> dict[str, Any]:
        raise NotImplementedError("NearDuplicates.result is yours to write")
