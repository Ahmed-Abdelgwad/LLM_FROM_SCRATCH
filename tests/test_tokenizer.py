import json
import random
from pathlib import Path

import pytest

from scratchllm.tokenizer.bpe import BASE_VOCAB, Pair, merge_word
from scratchllm.tokenizer.pretokenize import pretokenize
from scratchllm.tokenizer.tokenizer import Tokenizer

CORPUS = [
    "ذهب الولد إلى المدرسة صباحًا. ثم عاد الولد إلى البيت.",
    "مُحَمَّد ذهب إلى المدرسة ومُحَمَّد عاد",
    "The boy went to school. Then the boy came back home, and the school closed.",
    "call <phone> or write to <email> today\n\nNew paragraph here.",
] * 5
SPECIALS = ["<|endoftext|>", "<email>", "<phone>"]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    size = BASE_VOCAB + 120 + len(SPECIALS)
    return Tokenizer.train(CORPUS, vocab_size=size, special_tokens=SPECIALS)


def replay(merges: list[Pair], piece: bytes) -> list[int]:
    """The literal definition of encoding: apply merge 0 to the word, then merge 1, and so on."""
    ids = list(piece)
    for rank, pair in enumerate(merges):
        ids = merge_word(ids, pair, BASE_VOCAB + rank)
    return ids


# ---------- the id layout ----------


def test_ids_are_bytes_then_merges_then_specials(tok: Tokenizer):
    assert tok.vocab_size == BASE_VOCAB + 120 + len(SPECIALS)
    assert [tok.special_id(s) for s in SPECIALS] == [BASE_VOCAB + 120 + i for i in range(3)]
    assert tok.decode([65]) == "A"


def test_without_merges_every_byte_is_a_token():
    plain = Tokenizer([])
    assert plain.encode("aم") == [97, 0xD9, 0x85]
    assert plain.vocab_size == 256


# ---------- encoding is the replay of training ----------


def test_the_earliest_learned_merge_wins_not_the_leftmost_or_the_biggest():
    # ids: a=97 b=98 c=99. merge 0 joins (b,c), merge 1 joins (a,b). In "abc" both pairs are
    # present; the earlier one, (b,c), must go first, leaving "a" + "bc".
    plain = Tokenizer([(98, 99), (97, 98)])
    assert plain.encode("abc") == [97, 256]
    assert plain.encode("ab") == [257]


@pytest.mark.parametrize("seed", range(5))
def test_fast_encoding_equals_replaying_the_merges_in_order(tok: Tokenizer, seed: int):
    rng = random.Random(seed)
    alphabet = list("ابتثجحخدذر ولمنهي.\n,the school boy ") + ["ُ", "ّ", "😀"]
    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 30)))
        expected: list[int] = []
        for piece in pretokenize(text):
            expected.extend(replay(tok.merges, piece.encode("utf-8")))
        assert tok.encode(text, parse_special=False) == expected, repr(text)


def test_encoding_pieces_separately_gives_the_same_ids(tok: Tokenizer):
    text = "ذهب الولد إلى المدرسة"
    pieces = pretokenize(text)
    assert tok.encode(text) == [i for p in pieces for i in tok.encode(p)]


def test_training_text_gets_shorter_than_its_bytes(tok: Tokenizer):
    text = CORPUS[0]
    assert len(tok.encode(text)) < len(text.encode("utf-8")) * 0.7


# ---------- decoding ----------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello",
        "ذهب الولد إلى المدرسة",
        "مُحَمَّد",
        "سطر\nسطر آخر\n\nفقرة",
        "mixed عربي and English 123 ٣٤٥",
        "emoji 😀 and é and ﻻ",
        "text never seen in training: ζ ж 日本語",
    ],
)
def test_decode_of_encode_is_the_text(tok: Tokenizer, text: str):
    assert tok.decode(tok.encode(text)) == text


@pytest.mark.parametrize("seed", range(5))
def test_round_trip_on_random_unicode(tok: Tokenizer, seed: int):
    rng = random.Random(seed)
    alphabet = list("ab 1\n\t.,'!؟،ابمنُّ") + ["😀", "é", "日", "\u200f", "ﻻ", "\x00"]
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
        assert tok.decode(tok.encode(text)) == text, repr(text)


def test_a_fragment_of_a_character_is_replaced_or_refused(tok: Tokenizer):
    first_byte_of_meem = "م".encode()[0]
    assert tok.decode([first_byte_of_meem]) == "�"
    with pytest.raises(UnicodeDecodeError):
        tok.decode([first_byte_of_meem], errors="strict")


def test_an_id_outside_the_vocabulary_is_refused(tok: Tokenizer):
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])
    with pytest.raises(ValueError):
        tok.decode([-1])


# ---------- special tokens ----------


def test_special_tokens_become_their_reserved_id(tok: Tokenizer):
    ids = tok.encode("a<email>b<|endoftext|>")
    assert tok.special_id("<email>") in ids and tok.special_id("<|endoftext|>") == ids[-1]
    assert tok.decode(ids) == "a<email>b<|endoftext|>"


def test_special_tokens_are_plain_text_when_parsing_is_off(tok: Tokenizer):
    ids = tok.encode("<email>", parse_special=False)
    assert all(i < tok.special_id(SPECIALS[0]) for i in ids)
    assert tok.decode(ids) == "<email>"


def test_a_special_token_that_is_a_prefix_of_another_does_not_hide_it():
    plain = Tokenizer([], special_tokens=["<s>", "<s>x"])
    assert plain.encode("<s>x") == [plain.special_id("<s>x")]
    assert plain.encode("<s>y")[0] == plain.special_id("<s>")


def test_the_same_text_cached_or_not_gives_the_same_ids(tok: Tokenizer):
    first = tok.encode("ذهب الولد")
    assert tok.encode("ذهب الولد") == first


# ---------- files ----------


def test_save_and_load_keep_the_encoding(tok: Tokenizer, tmp_path: Path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == tok.merges and loaded.special_tokens == tok.special_tokens
    text = "ذهب الولد إلى <email> 😀"
    assert loaded.encode(text) == tok.encode(text)


def test_the_file_is_the_same_bytes_every_time(tok: Tokenizer, tmp_path: Path):
    tok.save(tmp_path / "a.json")
    tok.save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert not list(tmp_path.glob("*.tmp"))


def test_a_file_from_another_pretokenizer_is_refused(tok: Tokenizer, tmp_path: Path):
    path = tmp_path / "tok.json"
    tok.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pattern"] = r"\w+"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-tokenizer"):
        Tokenizer.load(path)


def test_a_file_of_another_kind_is_refused(tmp_path: Path):
    path = tmp_path / "x.json"
    path.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        Tokenizer.load(path)


# ---------- bad definitions ----------


@pytest.mark.parametrize(
    ("merges", "specials"),
    [
        ([(256, 97)], []),  # uses the id of the merge it is about to create
        ([(97, 98), (97, 98)], []),  # the same pair twice
        ([(97, 300)], []),  # an id that does not exist
        ([], ["<a>", "<a>"]),
        ([], [""]),
    ],
)
def test_inconsistent_definitions_are_refused(merges: list[Pair], specials: list[str]):
    with pytest.raises(ValueError):
        Tokenizer(merges, specials)


# ---------- one long training gives every smaller vocabulary ----------


def test_the_first_merges_of_a_longer_training_are_those_of_a_shorter_one(tok: Tokenizer):
    shorter = Tokenizer.train(
        CORPUS, vocab_size=BASE_VOCAB + 40 + len(SPECIALS), special_tokens=SPECIALS
    )
    assert tok.truncated(shorter.vocab_size).merges == shorter.merges


def test_truncated_tokenizer_encodes_like_a_trained_one(tok: Tokenizer):
    shorter = Tokenizer.train(
        CORPUS, vocab_size=BASE_VOCAB + 40 + len(SPECIALS), special_tokens=SPECIALS
    )
    small = tok.truncated(shorter.vocab_size)
    for text in CORPUS[:4]:
        assert small.encode(text) == shorter.encode(text)


def test_truncating_outside_the_possible_sizes_is_refused(tok: Tokenizer):
    with pytest.raises(ValueError):
        tok.truncated(tok.vocab_size + 1)
    with pytest.raises(ValueError):
        tok.truncated(BASE_VOCAB)  # no room for the special tokens
