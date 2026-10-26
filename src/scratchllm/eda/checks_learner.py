"""The three EDA checks that are yours to write.

Each is a contract, not a solution. Implement `update` and `result` so that `result()` returns
exactly the keys listed in the class docstring (the report and the later decisions read them).
Until a check is implemented its `update` raises NotImplementedError, and the runner reports it
as "not_implemented" instead of failing.

Suggested order: ScriptMix (smallest), SurfaceVariation, then NearDuplicates (needs the most
thinking). Write the tests in tests/test_eda_learner.py first, with a few hand-made strings
whose answers you know, then make them pass.
"""

from __future__ import annotations

from typing import Any


class ScriptMix:
    """EDA question #1 / acceptance criteria A3 and A4: which scripts do the letters use?

    For each document, count its *letters* (not digits, spaces or punctuation) and split them by
    script into Arabic, Latin and other. A document's share is letters-in-that-script divided by
    all its letters. A document with no letters at all is counted separately.

    result() keys:
      docs                         number of documents seen
      docs_without_letters         documents with no letters
      mean_arabic_share            average of the per-document Arabic share
      mean_latin_share             average of the per-document Latin share
      mean_other_share             average of the per-document share of every other script
      docs_arabic_share_ge_0_8     count with Arabic share >= 0.8       (A3, Arabic sources)
      docs_latin_share_ge_0_9      count with Latin share  >= 0.9       (A4, English sources)
      docs_mixed                   count with Arabic share >= 0.2 AND Latin share >= 0.2
    """

    name = "script_mix"

    def update(self, doc: dict[str, Any]) -> None:
        raise NotImplementedError("ScriptMix.update is yours to write")

    def result(self) -> dict[str, Any]:
        raise NotImplementedError("ScriptMix.result is yours to write")


class SurfaceVariation:
    """EDA question #4: how inconsistently is Arabic spelled? (decides the normalization rules)

    Count, over the whole sample, how often each variant appears, so we can decide which ones to
    unify. For every item below report two numbers: `docs_with` (documents containing it at least
    once) and `per_1000_letters` (occurrences per 1,000 Arabic letters in the sample).

    Items (result()["items"] maps each name to {"docs_with": int, "per_1000_letters": float}):
      diacritics_without_shadda    the short-vowel marks and tanween, EXCLUDING the shadda
      shadda                       reported on its own: dialect spelling uses it, so it must not
                                   be lumped with the other diacritics
      tatweel                      the elongation character
      alef_plain, alef_hamza_above, alef_hamza_below, alef_madda      the alef forms
      yeh, alef_maqsura            ي versus ى
      teh_marbuta_final, heh_final ة versus ه as the LAST letter of a word
      digits_ascii, digits_arabic_indic, digits_extended_arabic_indic

    Also return result()["arabic_letters"]: the total number of Arabic letters counted.

    Why it matters: the same word spelled two ways becomes two different tokens. If the rates
    are high and the variants are interchangeable, normalizing them lowers tokenizer fertility;
    if a variant is meaningful (shadda in dialects), normalizing would destroy information.
    """

    name = "surface_variation"

    def update(self, doc: dict[str, Any]) -> None:
        raise NotImplementedError("SurfaceVariation.update is yours to write")

    def result(self) -> dict[str, Any]:
        raise NotImplementedError("SurfaceVariation.result is yours to write")


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
