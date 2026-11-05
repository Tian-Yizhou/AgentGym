"""Tests for ``agentenv.envs.TextWorldExpressEnvClient`` against a live server.

These run in the *agent* environment (where ``agentenv`` is installed), not the server's:

    textworldexpress --host 127.0.0.1 --port 36005 &          # in the env-server conda env
    TWX_SERVER_URL=http://127.0.0.1:36005 pytest tests/test_client.py   # in the agent env

Parsing tests run without a server; the rest are skipped unless TWX_SERVER_URL is set.
"""

import os

import pytest
import requests

pytest.importorskip("agentenv")
from agentenv.envs import TextWorldExpressEnvClient  # noqa: E402

TEXTWORLDEXPRESS_TASK = "textworldexpress"
SERVER = os.environ.get("TWX_SERVER_URL")
needs_server = pytest.mark.skipif(not SERVER, reason="set TWX_SERVER_URL to a running textworldexpress server")

# data_idx of test-fold seed 0 for each suite (see games.py: suite_id * 3000 + 2 * 1000)
SUITE_TEST_IDX = {name: i * 3000 + 2000 for i, name in enumerate(
    ["cookingworld", "coin", "twc", "mapreader", "arithmetic", "sorting", "simonsays", "peckingorder"]
)}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Thought:\nI should look.\n\nAction:\nlook around", "look around"),
        ("Thought: go\nAction: move north\n", "move north"),
        ("Action:   take coin   ", "take coin"),
        ("Thought:\nhmm\n\nAction:\n\nopen fridge", "open fridge"),
    ],
)
def test_parse_action(text, expected):
    assert TextWorldExpressEnvClient.parse_action(text) == (expected, None)


@pytest.mark.parametrize("text", ["I will take the coin.", "Action:", "Action: a\nAction: b"])
def test_parse_action_errors(text):
    action, error = TextWorldExpressEnvClient.parse_action(text)
    assert action is None and error.startswith("Error:")


def as_llm_response(action):
    return f"Thought:\nI will do the next step of my plan.\n\nAction:\n{action}"


@needs_server
@pytest.mark.parametrize("suite", list(SUITE_TEST_IDX))
def test_full_episode_through_client(suite):
    client = TextWorldExpressEnvClient(env_server_base=SERVER, data_len=1, timeout=60)
    try:
        client.reset(SUITE_TEST_IDX[suite])
        first = client.observe()
        assert first.startswith(client.info["task_description"]) and "Available actions:" in first
        gold = requests.get(f"{SERVER}/gold_path", params={"id": client.env_id}, timeout=60).json()
        out = None
        for action in gold:
            out = client.step(as_llm_response(action))
            if out.done:
                break
        assert out.done and out.reward == 1.0
    finally:
        client.close()


@needs_server
def test_format_errors_keep_reward_and_do_not_advance():
    client = TextWorldExpressEnvClient(env_server_base=SERVER, data_len=1, timeout=60)
    try:
        client.reset(SUITE_TEST_IDX["mapreader"])
        gold = requests.get(f"{SERVER}/gold_path", params={"id": client.env_id}, timeout=60).json()
        out = None
        for action in gold:  # play until there is some partial credit
            out = client.step(as_llm_response(action))
            if 0 < out.reward < 1:
                break
        assert 0 < out.reward < 1
        moves = client.info["num_moves"]
        bad = client.step("Action: move north\nAction: move south")
        assert bad.state.startswith("Error:") and bad.reward == out.reward and not bad.done
        bad = client.step("no action here")
        assert bad.state.startswith("Error:") and bad.reward == out.reward and not bad.done
        detail = requests.get(f"{SERVER}/detail", params={"id": client.env_id}, timeout=60).json()
        assert detail["num_moves"] == moves
    finally:
        client.close()


@needs_server
def test_server_errors_surface_as_exceptions():
    client = TextWorldExpressEnvClient(env_server_base=SERVER, data_len=1, timeout=60)
    with pytest.raises(requests.exceptions.RequestException, match="400"):
        client.reset(10**9)
    client.close()
    with pytest.raises(requests.exceptions.RequestException, match="404"):
        client.reset(0)
