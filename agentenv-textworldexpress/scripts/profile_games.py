"""Profile every registered suite: horizon, action-list size and observation length.

Plays the TextWorldExpress gold path for a number of seeds through the server logic
(in-process, no HTTP) and reports, per suite:
  * gold path length (TWX gold agents random-walk, so this over-estimates the optimum),
  * number of valid actions per turn,
  * length of the formatted observation the LLM sees (characters, and tokens if a
    HuggingFace tokenizer path is given),
  * whether the gold path reaches done with reward 1.0.

Use it to size ``rounds`` / ``max_response_length`` for training, and whenever new suites
are added to ``games.py``.

Usage (in the env-server conda env):
    python scripts/profile_games.py --seeds 50 [--tokenizer AgentGym-RL/models/Qwen2.5-7B-Instruct]
"""

import argparse
import json
import statistics
import sys

from agentenv_textworldexpress.environment import ServerConfig, TextWorldExpressServer
from agentenv_textworldexpress.games import SUITES, encode


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--fold", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--tokenizer", default=None, help="optional HF tokenizer path for token counts")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a table")
    args = parser.parse_args()

    count_tokens = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        count_tokens = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731

    srv = TextWorldExpressServer(ServerConfig(step_limit=10_000))
    report = {}
    for suite in SUITES:
        gold_lens, n_actions, obs_len, obs_tok, wins = [], [], [], [], 0
        for k in range(args.seeds):
            env_id = srv.create()["id"]
            ep = srv.envs[env_id]
            payload = srv.reset(env_id, encode(suite.name, args.fold, k))
            # Gold path for exactly this game (regenerating the same seed gives the same game).
            ep.interface.generateNewGameJSON(payload["seed"], payload["fold"], True)
            gold = list(ep.interface.getGoldActionSequence())
            payload = srv.reset(env_id, payload["data_idx"])
            gold_lens.append(len(gold))
            observations = [payload["observation"]]
            n_actions.append(len(payload["valid_actions"]))
            for action in gold:
                if payload["done"]:
                    break
                payload = srv.step(env_id, action)
                observations.append(payload["observation"])
                n_actions.append(len(payload["valid_actions"]))
            wins += int(payload["done"] and payload["reward"] == 1.0)
            obs_len += [len(o) for o in observations]
            if count_tokens:
                obs_tok += [count_tokens(o) for o in observations]
            srv.close(env_id)
        row = {
            "gold_len_min_med_max": [min(gold_lens), statistics.median(gold_lens), max(gold_lens)],
            "gold_len_p90": pct(gold_lens, 0.9),
            "valid_actions_med_max": [statistics.median(n_actions), max(n_actions)],
            "obs_chars_med_p90": [statistics.median(obs_len), pct(obs_len, 0.9)],
            "gold_reaches_reward_1": f"{wins}/{args.seeds}",
        }
        if count_tokens:
            row["obs_tokens_med_p90_max"] = [statistics.median(obs_tok), pct(obs_tok, 0.9), max(obs_tok)]
        report[suite.name] = row
        print(f"profiled {suite.name}", file=sys.stderr)
    srv.shutdown()

    if args.json:
        print(json.dumps(report, indent=2))
        return
    cols = list(next(iter(report.values())).keys())
    print("| suite | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, row in report.items():
        print(f"| {name} | " + " | ".join(str(row[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
