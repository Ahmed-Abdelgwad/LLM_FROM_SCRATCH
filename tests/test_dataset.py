from pathlib import Path

import numpy as np
import pytest
import torch

from scratchllm.data.tokens import DTYPE
from scratchllm.training.dataset import TokenWindows, window_count


def token_file(path: Path, ids: list[int]) -> Path:
    np.asarray(ids, dtype=DTYPE).tofile(path)
    return path


@pytest.mark.parametrize(
    ("n", "t", "stride", "expected"),
    [(10, 4, 4, 2), (9, 4, 4, 2), (8, 4, 4, 1), (5, 4, 4, 1), (4, 4, 4, 0), (0, 4, 4, 0),
     (10, 4, 1, 6), (10, 4, 2, 3)],
)  # fmt: skip
def test_window_count(n: int, t: int, stride: int, expected: int):
    assert window_count(n, t, stride) == expected


def test_target_is_the_input_shifted_by_one(tmp_path: Path):
    data = TokenWindows([token_file(tmp_path / "a.bin", list(range(100, 111)))], 4)
    x, y = data[0]
    assert x.tolist() == [100, 101, 102, 103] and y.tolist() == [101, 102, 103, 104]
    x, y = data[1]
    assert x.tolist() == [104, 105, 106, 107] and y.tolist() == [105, 106, 107, 108]
    assert len(data) == 2  # 11 ids: the third window would need ids 108..112
    assert x.dtype == torch.int64 and y.shape == (4,)


def test_stride_moves_the_windows(tmp_path: Path):
    data = TokenWindows([token_file(tmp_path / "a.bin", list(range(20)))], 4, stride=2)
    assert [data[i][0][0].item() for i in range(len(data))] == list(range(0, 15, 2))


def test_windows_never_cross_from_one_file_into_the_next(tmp_path: Path):
    a = token_file(tmp_path / "a.bin", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = token_file(tmp_path / "b.bin", [101, 102, 103, 104, 105, 106])
    data = TokenWindows([a, b], 4)
    windows = [(x.tolist(), y.tolist()) for x, y in (data[i] for i in range(len(data)))]
    assert windows == [
        ([1, 2, 3, 4], [2, 3, 4, 5]),
        ([5, 6, 7, 8], [6, 7, 8, 9]),
        ([101, 102, 103, 104], [102, 103, 104, 105]),
    ]


def test_every_window_stays_inside_its_file(tmp_path: Path):
    files = [token_file(tmp_path / f"{i}.bin", list(range(i * 1000, i * 1000 + n)))
             for i, n in enumerate([13, 50, 7, 31])]  # fmt: skip
    data = TokenWindows(files, 5, stride=3)
    for i in range(len(data)):
        x, y = data[i]
        assert x[0] // 1000 == y[-1] // 1000  # same file
        assert torch.equal(x[1:], y[:-1])


def test_empty_and_too_short_files_give_no_windows(tmp_path: Path):
    empty = token_file(tmp_path / "e.bin", [])
    short = token_file(tmp_path / "s.bin", [1, 2, 3])
    assert len(TokenWindows([empty, short], 4)) == 0
    assert len(TokenWindows([], 4)) == 0
    both = TokenWindows([empty, short, token_file(tmp_path / "l.bin", list(range(10)))], 4)
    assert len(both) == 2 and both[0][0].tolist() == [0, 1, 2, 3]


def test_out_of_range_indices_are_refused(tmp_path: Path):
    data = TokenWindows([token_file(tmp_path / "a.bin", list(range(10)))], 4)
    for bad in (-1, len(data)):
        with pytest.raises(IndexError):
            data[bad]


def test_bad_sizes_are_refused(tmp_path: Path):
    f = token_file(tmp_path / "a.bin", [1, 2, 3])
    with pytest.raises(ValueError):
        TokenWindows([f], 0)
    with pytest.raises(ValueError):
        TokenWindows([f], 2, stride=0)


def test_ids_above_32767_are_not_turned_negative(tmp_path: Path):
    data = TokenWindows([token_file(tmp_path / "a.bin", [65535, 40000, 32768, 1, 2])], 3)
    x, y = data[0]
    assert x.tolist() == [65535, 40000, 32768] and y.tolist() == [40000, 32768, 1]
