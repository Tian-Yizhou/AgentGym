# Agent Environments - TextWorldExpress

AgentGym environment server for [TextWorldExpress](https://github.com/cognitiveailab/TextWorldExpress)
(`textworld-express==1.1.0`). One server exposes all eight TextWorldExpress games with
their **default parameters**:

| suite | game | task |
|---|---|---|
| `cookingworld` | CookingWorld | find ingredients, follow the cookbook recipe, prepare and eat the meal |
| `coin` | Coin Collector | explore the house and take the coin |
| `twc` | TextWorld Commonsense | put objects in their usual locations |
| `mapreader` | Map Reader | use a map to fetch a coin and put it in the box |
| `arithmetic` | Arithmetic | solve a math problem, put the matching object in the box |
| `sorting` | Sorting | put objects in the box in order of increasing quantity |
| `simonsays` | Simon Says | do exactly what you are told |
| `peckingorder` | Pecking Order | follow the instructions found in the environment |

## Setup

TextWorldExpress needs Java >= 1.8. `setup.sh` installs OpenJDK from conda-forge into the env:

```sh
bash setup.sh        # creates conda env `agentenv-textworldexpress` and installs this package
```

or by hand:

```sh
conda create -n agentenv-textworldexpress -c conda-forge python=3.10 pip openjdk=17 -y
conda activate agentenv-textworldexpress
pip install -e .
```

## Launch

```sh
textworldexpress --host 0.0.0.0 --port 36005
```

| flag (env var) | default | meaning |
|---|---|---|
| `--reward-mode` (`TWX_REWARD_MODE`) | `score` | `score`: TextWorldExpress's normalized score (1.0 = solved, partial credit in between, negative when an arithmetic/sorting/simonsays/peckingorder episode fails). `success`: 1.0 if solved, else 0.0 |
| `--show-valid-actions` / `--no-show-valid-actions` (`TWX_SHOW_VALID_ACTIONS`) | on | append the list of valid actions to every observation |
| `--step-limit` (`TWX_STEP_LIMIT`) | 100 | episode ends after this many steps (TWX default); keep it >= the RL rounds |
| `--num-jvms` (`TWX_NUM_JVMS`) | 1 | Java processes shared by all environments |
| `--jvm-heap` (`TWX_JVM_HEAP`) | `2g` | `-Xmx` per shared Java process |

Run a single uvicorn worker (the default): environment ids live in the server process.

## Tasks and item ids

`data_idx` (the number in item ids such as `textworldexpress_2005`) encodes game, fold and seed:

```
data_idx = suite_id * 3000 + fold_id * 1000 + k        fold_id: train=0, dev=1, test=2
seed     = {train: 0, dev: 10000, test: 20000}[fold] + k
```

Suites are listed in `agentenv_textworldexpress/games.py` (append-only: never reorder or
change existing entries, since that silently changes what existing item ids mean). Train,
dev and test use TextWorldExpress's own disjoint seed folds, and also its fold-specific
object lists.

Generate the AgentGym-RL data files (any Python, no Java needed):

```sh
python AgentGym/agentenv-textworldexpress/scripts/make_data.py --out-dir AgentGym-RL
# -> AgentItemId/train/textworldexpress_train.json, AgentItemId/eval/textworldexpress_test.json,
#    AgentEval/textworldexpress/ (test fold) and AgentEval/textworldexpress_dev/ (dev fold),
#    each with one category file per game for main_generation's per-game breakdown.
```

## API

| endpoint | body / query | returns |
|---|---|---|
| `POST /create` | | `{"id": int}` |
| `POST /reset` | `{"id", "data_idx"}` | payload (below) |
| `POST /step` | `{"id", "action"}` | payload, plus `action` and `valid_action` |
| `GET /observation` | `?id=` | current observation text |
| `GET /detail` | `?id=` | last payload |
| `GET /gold_path` | `?id=` | TextWorldExpress's reference solution for the current game (debugging/SFT; random-walk based, not shortest) |
| `POST /close` | `{"id"}` | `true` |

Payload: `observation` (text for the LLM: task description on reset, game text, available
actions), `reward` (cumulative, per `--reward-mode`), `score`, `done`, `success`,
`failure`, `num_moves`, `valid_actions`, `raw_observation`, `task_description`,
`data_idx`, `suite`, `game`, `fold`, `seed`.

Actions are matched exactly against the valid actions after light normalization (case,
whitespace, surrounding quotes, trailing period). Anything else is an invalid move: TWX
answers "Unknown action", the score is unchanged and the move counts toward the step limit.
Errors are HTTP errors (`400` bad `data_idx`, `404` unknown id, `409` step before reset or
after the episode ended).

## Implementation notes

- All environments share a small pool of JVMs (one `PythonInterface` object each), instead of
  one Java process per environment as in the stock Python wrapper. TextWorldExpress keeps game
  state per `PythonInterface`, so this is safe and much cheaper for RL's many short-lived envs.
- Exception: `sorting` keeps JVM-global state (its problem objects are shared by all games
  built from the same problem), so two sorting games in one JVM corrupt each other. Each env
  that plays sorting gets its own JVM (~0.7 s on its first sorting reset).
  `tests/test_server.py::test_concurrent_same_game_isolation` checks every game for this.

## Tests

```sh
pytest tests/test_games.py tests/test_server.py          # in this env (needs Java)
# client tests run in the agent env (where `agentenv` is installed), against a live server:
TWX_SERVER_URL=http://127.0.0.1:36005 pytest tests/test_client.py
python scripts/profile_games.py --seeds 50                # horizon / action-list / observation sizes
```
