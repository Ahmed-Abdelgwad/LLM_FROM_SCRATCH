"""The three EDA checks about Arabic text.

Each is a contract: `result()` returns exactly the keys listed in the class docstring (the
report and the later decisions read them). Until a check is implemented its `update` raises
NotImplementedError, and the runner reports it as "not_implemented" instead of failing.

Status: all three are implemented (written by the assistant at the owner's request, with the
decisions explained in their docstrings).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import zlib
from collections import Counter
from typing import Any

import numpy as np

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


# MinHash works on hash values below a prime larger than 2**32. With a_i < 2**32 and x < 2**32,
# a_i * x + b_i < 2**64, so the arithmetic below cannot overflow unsigned 64-bit integers.
_PRIME = 4_294_967_311


def word_shingles(text: str, size: int) -> list[str]:
    """Overlapping runs of `size` words. A text shorter than `size` words is one shingle."""
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    return [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]


class NearDuplicates:
    """EDA question #3 / acceptance criterion A2: how much of the sample is near-duplicate?

    Two documents are near-duplicates when the Jaccard similarity of their sets of word
    shingles is >= `threshold`. Exact duplicates were already removed at sampling time, so this
    finds templated pages (same text with a few words changed).

    How it works (MinHash + LSH banding, no extra dependency):
    1. A document becomes the set of its word shingles (`word_shingles`), each hashed to 32 bits
       with crc32. crc32 is used instead of hash() on purpose: Python randomises hash() for str
       between runs, which would make the result differ every time.
    2. A signature of `num_perm` numbers summarises that set: for each of `num_perm` random hash
       functions (a*x+b mod prime), the minimum over the shingles. Two signatures agree at a
       position with probability equal to the Jaccard similarity of the sets, so the fraction of
       agreeing positions estimates it (standard error about 0.03 at 128 permutations).
    3. LSH banding finds candidates without comparing all pairs: the signature is cut into
       bands of `rows_per_band` positions, and documents whose band is identical share a bucket.
       With 128 permutations and 8 rows per band, a pair at similarity 0.8 shares a bucket with
       probability about 0.95, and at 0.9 about 0.9999, while pairs far below the threshold
       rarely do.
    4. A candidate is confirmed when its estimated similarity is >= `threshold`. Confirmed pairs
       are merged into clusters (union-find), so a template family counts as one cluster.

    Approximation, deliberately: inside a bucket each member is compared with the bucket's first
    member only, so the cost stays linear even when thousands of pages share a template. A pair
    can therefore be missed by one band and still be joined through another band or through the
    cluster. Recall is high but not exact; the estimate of similarity is also noisy near the
    threshold, so a rate should be read as "about", not to the unit.

    result() keys:
      docs                          number of documents seen
      docs_without_signature        documents with no words, which cannot be compared
      threshold, num_perm, shingle_words, rows_per_band   the parameters actually used
      signatures_sha256             fingerprint of all signatures (16 hex of SHA-256); equal inputs
                                    and parameters must give it on every run and machine
      near_duplicate_docs           documents that have at least one near-duplicate neighbour
      near_duplicate_rate           near_duplicate_docs / docs        (A2 asks for <= 0.30)
      clusters                      number of groups of 2 or more near-duplicates
      largest_cluster_sizes         sizes of the 10 largest clusters, descending
      largest_cluster_examples      for each of those: size, the first 3 document ids and the first
                                    120 characters of the first document, so a human can read what
                                    the template is
    """

    name = "near_duplicates"

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        shingle_words: int = 5,
        rows_per_band: int = 8,
        seed: int = 0,
    ) -> None:
        if num_perm % rows_per_band:
            raise ValueError("num_perm must be a multiple of rows_per_band")
        self.threshold, self.num_perm = threshold, num_perm
        self.shingle_words, self.rows_per_band = shingle_words, rows_per_band
        rng = np.random.RandomState(seed)  # fixed generator, stable across numpy versions
        self._a = rng.randint(1, 2**32 - 1, size=num_perm, dtype=np.uint64)[:, None]
        self._b = rng.randint(0, 2**32 - 1, size=num_perm, dtype=np.uint64)[:, None]
        self.docs = 0
        self.docs_without_signature = 0
        self._ids: list[str] = []
        self._snippets: list[str] = []
        self._signatures: list[Any] = []

    def update(self, doc: dict[str, Any]) -> None:
        self.docs += 1
        shingles = word_shingles(doc["text"], self.shingle_words)
        if not shingles:
            self.docs_without_signature += 1
            return
        hashed = np.unique(
            np.fromiter((zlib.crc32(s.encode("utf-8")) for s in shingles), np.uint64)
        )
        # row i = hash function i applied to every shingle; the signature keeps each row's minimum
        self._signatures.append(((self._a * hashed + self._b) % _PRIME).min(axis=1))
        self._ids.append(str(doc["id"]))
        self._snippets.append(doc["text"][:120])

    def _clusters(self) -> list[list[int]]:
        n = len(self._signatures)
        if n < 2:
            return []
        sigs = np.stack(self._signatures)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]  # path halving
                i = parent[i]
            return i

        for band in range(self.num_perm // self.rows_per_band):
            columns = sigs[:, band * self.rows_per_band : (band + 1) * self.rows_per_band]
            first_in_bucket: dict[bytes, int] = {}
            for i in range(n):
                first = first_in_bucket.setdefault(columns[i].tobytes(), i)
                if first == i or find(first) == find(i):
                    continue
                if (sigs[first] == sigs[i]).mean() >= self.threshold:
                    parent[find(i)] = find(first)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return [members for members in groups.values() if len(members) >= 2]

    def _fingerprint(self) -> str:
        if not self._signatures:
            return ""
        return hashlib.sha256(np.stack(self._signatures).tobytes()).hexdigest()[:16]

    def result(self) -> dict[str, Any]:
        clusters = sorted(self._clusters(), key=lambda members: (-len(members), members[0]))
        top = clusters[:10]
        near_duplicate_docs = sum(len(members) for members in clusters)
        return {
            "docs": self.docs,
            "docs_without_signature": self.docs_without_signature,
            "threshold": self.threshold,
            "num_perm": self.num_perm,
            "shingle_words": self.shingle_words,
            "rows_per_band": self.rows_per_band,
            "signatures_sha256": self._fingerprint(),
            "near_duplicate_docs": near_duplicate_docs,
            "near_duplicate_rate": round(near_duplicate_docs / self.docs, 4) if self.docs else 0.0,
            "clusters": len(clusters),
            "largest_cluster_sizes": [len(members) for members in top],
            "largest_cluster_examples": [
                {
                    "size": len(members),
                    "ids": [self._ids[i] for i in members[:3]],
                    "first_text": self._snippets[members[0]],
                }
                for members in top
            ],
        }
