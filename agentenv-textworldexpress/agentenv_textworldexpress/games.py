"""Game suites and the ``data_idx`` codec for the TextWorldExpress env server.

This module is pure Python (no TextWorldExpress / Java / FastAPI import), so data
generation scripts and tests can use it from any environment.

``data_idx`` layout
-------------------
::

    data_idx = suite_id * SUITE_BLOCK + fold_id * FOLD_BLOCK + k      (0 <= k < FOLD_BLOCK)
    seed     = FOLD_SEED_BASE[fold] + k

TextWorldExpress defines exactly 1000 seeds per fold (train 0-999, dev 10000-10999,
test 20000-20999), so ``FOLD_BLOCK = 1000`` and ``SUITE_BLOCK = 3000``. The fold is
passed to TextWorldExpress as well, because some games (twc, cookingworld) also use
different object/ingredient lists per fold.

The registry is APPEND-ONLY: never reorder, rename, or re-parameterise an existing
suite, because item ids already written to data files would silently change meaning.
Add new suites (e.g. harder parameter settings) at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

FOLDS = ("train", "dev", "test")
FOLD_SEED_BASE = {"train": 0, "dev": 10000, "test": 20000}
FOLD_BLOCK = 1000
SUITE_BLOCK = FOLD_BLOCK * len(FOLDS)


@dataclass(frozen=True)
class Suite:
    name: str  # used in eval category file names (``<name>.json``); no "." allowed
    game: str  # TextWorldExpress ``gameName``
    params: str = ""  # TextWorldExpress ``gameParams``; "" means the TextWorldExpress defaults


# Every TextWorldExpress game with its default parameters, in the order of
# ``textworld_express.GAME_NAMES``. The defaults (from GameGenerator.scala, TWX 1.1.0):
#   cookingworld: numLocations=11, numIngredients=3, numDistractorItems=10, includeDoors=1, limitInventorySize=1
#   coin:         numLocations=11, numDistractorItems=0, includeDoors=1, limitInventorySize=1
#   twc:          numLocations=3, numItemsToPutAway=4, includeDoors=0, limitInventorySize=0
#   mapreader:    numLocations=15, maxDistanceApart=4, maxDistractorItemsPerLocation=3, includeDoors=0, limitInventorySize=0
#   arithmetic, sorting, peckingorder: no parameters
#   simonsays:    gameLength=5, numDistractors=3, memorization=0, verbose=0
SUITES: tuple[Suite, ...] = (
    Suite("cookingworld", "cookingworld"),
    Suite("coin", "coin"),
    Suite("twc", "twc"),
    Suite("mapreader", "mapreader"),
    Suite("arithmetic", "arithmetic"),
    Suite("sorting", "sorting"),
    Suite("simonsays", "simonsays"),
    Suite("peckingorder", "peckingorder"),
)

SUITE_IDS = {suite.name: i for i, suite in enumerate(SUITES)}


@dataclass(frozen=True)
class GameSpec:
    """A concrete, reproducible game instance."""

    data_idx: int
    suite: Suite
    fold: str
    seed: int


def decode(data_idx: int) -> GameSpec:
    """Map a ``data_idx`` to the game it denotes. Raises ``ValueError`` if out of range."""
    if not isinstance(data_idx, int) or isinstance(data_idx, bool) or data_idx < 0:
        raise ValueError(f"data_idx must be a non-negative int, got {data_idx!r}")
    suite_id, rest = divmod(data_idx, SUITE_BLOCK)
    if suite_id >= len(SUITES):
        raise ValueError(
            f"data_idx {data_idx} refers to suite #{suite_id}, but only {len(SUITES)} suites "
            f"are registered (valid data_idx range: 0..{len(SUITES) * SUITE_BLOCK - 1})"
        )
    fold_id, k = divmod(rest, FOLD_BLOCK)
    fold = FOLDS[fold_id]
    return GameSpec(data_idx=data_idx, suite=SUITES[suite_id], fold=fold, seed=FOLD_SEED_BASE[fold] + k)


def encode(suite: str, fold: str, k: int) -> int:
    """Inverse of :func:`decode`: the ``data_idx`` of the ``k``-th seed of ``fold`` in ``suite``."""
    if suite not in SUITE_IDS:
        raise ValueError(f"unknown suite {suite!r}; known: {', '.join(SUITE_IDS)}")
    if fold not in FOLD_SEED_BASE:
        raise ValueError(f"unknown fold {fold!r}; known: {', '.join(FOLDS)}")
    if not 0 <= k < FOLD_BLOCK:
        raise ValueError(f"k must be in [0, {FOLD_BLOCK}), got {k}")
    return SUITE_IDS[suite] * SUITE_BLOCK + FOLDS.index(fold) * FOLD_BLOCK + k


def iter_data_idx(suite: str, fold: str, n: int, start: int = 0) -> Iterator[int]:
    """The ``data_idx`` values of seeds ``start .. start+n-1`` of ``fold`` in ``suite``."""
    for k in range(start, start + n):
        yield encode(suite, fold, k)
