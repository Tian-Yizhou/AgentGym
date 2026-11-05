"""Write the item-id files AgentGym-RL needs for the ``textworldexpress`` task.

Runs in any Python >= 3.8 environment (only needs ``games.py``, no TextWorldExpress/Java).

Outputs, relative to ``--out-dir`` (the ``AgentGym-RL/`` directory of AgentGym-RL):

    AgentItemId/train/textworldexpress_train.json        training items   (train fold)
    AgentItemId/eval/textworldexpress_test.json          test items       (test fold)
    AgentEval/textworldexpress/textworldexpress_test.json + <suite>.json
                                                         eval dir for main_generation (test fold,
                                                         one category file per suite)
    AgentEval/textworldexpress_dev/textworldexpress_test.json + <suite>.json
                                                         same layout for the dev fold (model selection)

Each item is ``{"item_id": "textworldexpress_<data_idx>"}``; see ``games.py`` for the layout.
Item k of a suite/fold uses the k-th seed of that TextWorldExpress fold, so train, dev and
test never share a game.

Example (from the repo root):
    python AgentGym/agentenv-textworldexpress/scripts/make_data.py --out-dir AgentGym-RL
"""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

TASK_NAME = "textworldexpress"

# Load games.py by path so this script does not need the server package (or Java) installed.
_GAMES_PY = Path(__file__).resolve().parents[1] / "agentenv_textworldexpress" / "games.py"
_spec = importlib.util.spec_from_file_location("twx_games", _GAMES_PY)
games = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = games  # dataclasses look the module up while it executes
_spec.loader.exec_module(games)


def item(data_idx: int) -> dict:
    return {"item_id": f"{TASK_NAME}_{data_idx}"}


def write_json(path: Path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2) + "\n")
    print(f"wrote {len(items):5d} items -> {path}")


def write_eval_dir(directory: Path, per_suite: dict) -> None:
    """main_generation layout: <task>_test.json plus one category file per suite, nothing else."""
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.json"):
        stale.unlink()
    all_items = [item(i) for ids in per_suite.values() for i in ids]
    write_json(directory / f"{TASK_NAME}_test.json", all_items)
    for suite, ids in per_suite.items():
        write_json(directory / f"{suite}.json", [item(i) for i in ids])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, type=Path, help="the AgentGym-RL/ directory")
    parser.add_argument("--suites", nargs="+", default=[s.name for s in games.SUITES],
                        help="suites to include (default: all)")
    parser.add_argument("--n-train", type=int, default=200, help="train seeds per suite (default 200)")
    parser.add_argument("--n-dev", type=int, default=50, help="dev seeds per suite (default 50)")
    parser.add_argument("--n-test", type=int, default=50, help="test seeds per suite (default 50)")
    parser.add_argument("--shuffle-seed", type=int, default=0,
                        help="seed for the order of the training file (the RL dataloader shuffles too)")
    args = parser.parse_args()

    for suite in args.suites:
        if suite not in games.SUITE_IDS:
            parser.error(f"unknown suite {suite!r}; known: {', '.join(games.SUITE_IDS)}")
    for n in (args.n_train, args.n_dev, args.n_test):
        if not 0 <= n <= games.FOLD_BLOCK:
            parser.error(f"seed counts must be in [0, {games.FOLD_BLOCK}]")

    splits = {
        fold: {s: list(games.iter_data_idx(s, fold, n)) for s in args.suites}
        for fold, n in (("train", args.n_train), ("dev", args.n_dev), ("test", args.n_test))
    }

    # Invariants: the folds are disjoint, and every item decodes back to its own suite/fold.
    seen = {}
    for fold, per_suite in splits.items():
        for suite, ids in per_suite.items():
            for i in ids:
                spec = games.decode(i)
                assert (spec.suite.name, spec.fold) == (suite, fold), (i, spec)
                assert i not in seen, f"data_idx {i} is in both {seen[i]} and {fold}"
                seen[i] = fold

    train = [item(i) for ids in splits["train"].values() for i in ids]
    random.Random(args.shuffle_seed).shuffle(train)
    out = args.out_dir
    write_json(out / "AgentItemId" / "train" / f"{TASK_NAME}_train.json", train)
    write_json(out / "AgentItemId" / "eval" / f"{TASK_NAME}_test.json",
               [item(i) for ids in splits["test"].values() for i in ids])
    write_eval_dir(out / "AgentEval" / TASK_NAME, splits["test"])
    write_eval_dir(out / "AgentEval" / f"{TASK_NAME}_dev", splits["dev"])


if __name__ == "__main__":
    main()
