"""Tests for the suite registry and data_idx codec (pure Python, no Java needed)."""

import pytest

from agentenv_textworldexpress.games import (
    FOLD_BLOCK,
    FOLD_SEED_BASE,
    FOLDS,
    SUITE_BLOCK,
    SUITES,
    decode,
    encode,
    iter_data_idx,
)


def test_roundtrip_and_uniqueness():
    seen = set()
    for suite in SUITES:
        for fold in FOLDS:
            for k in (0, 1, 499, FOLD_BLOCK - 1):
                idx = encode(suite.name, fold, k)
                assert idx not in seen
                seen.add(idx)
                spec = decode(idx)
                assert (spec.suite, spec.fold, spec.seed) == (suite, fold, FOLD_SEED_BASE[fold] + k)


def test_seeds_match_textworldexpress_folds():
    # TWX: train 0-999, dev 10000-10999, test 20000-20999
    assert decode(encode("twc", "train", 999)).seed == 999
    assert decode(encode("twc", "dev", 0)).seed == 10000
    assert decode(encode("twc", "test", 999)).seed == 20999


def test_layout_is_stable():
    # Item ids written to data files depend on this layout; changing it is a breaking change.
    assert SUITE_BLOCK == 3000 and FOLD_BLOCK == 1000
    assert [s.name for s in SUITES[:8]] == [
        "cookingworld", "coin", "twc", "mapreader", "arithmetic", "sorting", "simonsays", "peckingorder",
    ]
    assert encode("cookingworld", "train", 5) == 5
    assert encode("twc", "test", 0) == 2 * 3000 + 2 * 1000


def test_suite_names_are_safe_for_files():
    names = [s.name for s in SUITES]
    assert len(set(names)) == len(names)
    for name in names:
        assert "." not in name and "/" not in name
        assert not name.startswith("textworldexpress_test")  # would collide with main_generation's test file


@pytest.mark.parametrize("bad", [-1, len(SUITES) * SUITE_BLOCK, 10**9, 1.5, "3", True])
def test_decode_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        decode(bad)


def test_encode_rejects_bad_input():
    with pytest.raises(ValueError):
        encode("nope", "train", 0)
    with pytest.raises(ValueError):
        encode("twc", "val", 0)
    with pytest.raises(ValueError):
        encode("twc", "train", FOLD_BLOCK)


def test_iter_data_idx():
    assert list(iter_data_idx("coin", "test", 3, start=10)) == [encode("coin", "test", k) for k in (10, 11, 12)]
