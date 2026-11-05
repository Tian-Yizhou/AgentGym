"""End-to-end tests of the HTTP server (needs Java on PATH; CPU only)."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from agentenv_textworldexpress import app
from agentenv_textworldexpress.games import SUITES, encode


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan -> starts the JVM pool
        yield c


@pytest.fixture(scope="module")
def server(client):
    return client.app.state.server


def new_env(client):
    res = client.post("/create")
    assert res.status_code == 200
    return res.json()["id"]


def reset(client, env_id, data_idx):
    res = client.post("/reset", json={"id": env_id, "data_idx": data_idx})
    assert res.status_code == 200, res.text
    return res.json()


def step(client, env_id, action):
    res = client.post("/step", json={"id": env_id, "action": action})
    assert res.status_code == 200, res.text
    return res.json()


def gold_path(server, env_id, payload):
    """TWX gold path for exactly the game currently loaded in env_id (does not change its state)."""
    iface = server.envs[env_id].interface
    iface.generateNewGameJSON(payload["seed"], payload["fold"], True)
    gold = list(iface.getGoldActionSequence())
    iface.generateNewGameJSON(payload["seed"], payload["fold"], False)  # restore the identical fresh game
    return gold


def test_hello(client):
    assert "TextWorldExpress" in client.get("/").json()


def test_reset_payload_and_observation_format(client):
    env_id = new_env(client)
    p = reset(client, env_id, encode("twc", "train", 0))
    assert (p["suite"], p["fold"], p["seed"], p["data_idx"]) == ("twc", "train", 0, encode("twc", "train", 0))
    assert (p["reward"], p["score"], p["done"], p["num_moves"]) == (0.0, 0.0, False, 0)
    # The goal is shown to the model, followed by the room description and the full sorted action list.
    assert p["observation"].startswith(p["task_description"])
    assert p["raw_observation"].strip() in p["observation"]
    assert p["valid_actions"] == sorted(p["valid_actions"]) and p["valid_actions"]
    assert p["observation"].endswith("Available actions: " + json.dumps(p["valid_actions"]))
    assert client.get("/observation", params={"id": env_id}).json() == p["observation"]
    assert client.get("/detail", params={"id": env_id}).json()["data_idx"] == p["data_idx"]


@pytest.mark.parametrize("suite", [s.name for s in SUITES])
def test_gold_path_solves_every_suite(client, server, suite):
    env_id = new_env(client)
    p = reset(client, env_id, encode(suite, "test", 3))
    for action in gold_path(server, env_id, p):
        if p["done"]:  # random-walk gold paths may keep wandering after the task is solved
            break
        p = step(client, env_id, action)
        assert p["valid_action"], action
    assert p["done"] and p["success"] and p["reward"] == 1.0 and p["score"] == 1.0
    assert "Available actions" not in p["observation"]  # no action list once the episode is over
    # Stepping a finished episode is an explicit error, not a silent 200.
    res = client.post("/step", json={"id": env_id, "action": "look around"})
    assert res.status_code == 409
    client.post("/close", json={"id": env_id})


def test_gold_path_endpoint_does_not_disturb_episode(client):
    env_id = new_env(client)
    p = reset(client, env_id, encode("arithmetic", "train", 4))
    step(client, env_id, "look around")
    gold = client.get("/gold_path", params={"id": env_id}).json()
    assert gold and all(isinstance(a, str) for a in gold)
    assert client.get("/detail", params={"id": env_id}).json()["num_moves"] == 1
    # Replaying the gold path from a fresh reset solves the game.
    p = reset(client, env_id, p["data_idx"])
    for action in gold:
        if p["done"]:
            break
        p = step(client, env_id, action)
    assert p["success"]


@pytest.mark.parametrize("suite", ["arithmetic", "sorting", "simonsays", "peckingorder"])
def test_in_game_failure_is_a_normal_response(client, server, suite):
    """These games end a failed episode with a negative score; that is an outcome, not an error."""
    failures = 0
    for k in range(5):
        env_id = new_env(client)
        p = reset(client, env_id, encode(suite, "train", k))
        gold = client.get("/gold_path", params={"id": env_id}).json()
        for t in range(60):
            if p["done"]:
                break
            # deliberately pick a game action that is not the reference next step
            wrong = [a for a in p["valid_actions"] if a not in ("look around", "inventory")
                     and (t >= len(gold) or a != gold[t])]
            p = step(client, env_id, (wrong or p["valid_actions"])[0])
        if p["failure"]:
            failures += 1
            assert p["done"] and not p["success"] and p["reward"] < 0 and p["score"] < 0
        client.post("/close", json={"id": env_id})
    assert failures > 0


def test_invalid_action_keeps_score_and_counts_a_move(client):
    env_id = new_env(client)
    p0 = reset(client, env_id, encode("coin", "train", 1))
    p = step(client, env_id, "dance wildly")
    assert not p["valid_action"] and not p["done"]
    assert p["raw_observation"] == "Unknown action: I'm not sure what you mean."
    assert p["score"] == p0["score"] and p["num_moves"] == 1
    assert p["valid_actions"] == p0["valid_actions"]


@pytest.mark.parametrize("variant", ["LOOK AROUND", "  look   around. ", "`look around`", '"Look around."'])
def test_action_normalization(client, variant):
    env_id = new_env(client)
    reset(client, env_id, encode("twc", "train", 2))
    p = step(client, env_id, variant)
    assert p["valid_action"] and p["action"] == "look around"


def test_no_fuzzy_matching(client):
    env_id = new_env(client)
    reset(client, env_id, encode("twc", "train", 2))
    assert not step(client, env_id, "look")["valid_action"]
    assert not step(client, env_id, "I will look around now")["valid_action"]


def test_same_data_idx_is_deterministic_across_envs(client):
    idx = encode("cookingworld", "train", 7)
    a, b = new_env(client), new_env(client)
    pa, pb = reset(client, a, idx), reset(client, b, idx)
    assert pa["observation"] == pb["observation"]
    for action in pa["valid_actions"][:3]:
        assert step(client, a, action)["observation"] == step(client, b, action)["observation"]


def test_reset_switches_suites_and_restarts_episode(client):
    env_id = new_env(client)
    p1 = reset(client, env_id, encode("twc", "train", 0))
    step(client, env_id, "look around")
    p2 = reset(client, env_id, encode("coin", "train", 0))
    assert p2["suite"] == "coin" and p2["num_moves"] == 0 and p2["observation"] != p1["observation"]
    p3 = reset(client, env_id, encode("twc", "train", 0))
    assert p3["observation"] == p1["observation"]


def test_folds_produce_different_games(client):
    env_id = new_env(client)
    train = reset(client, env_id, encode("twc", "train", 0))["observation"]
    test = reset(client, env_id, encode("twc", "test", 0))["observation"]
    assert train != test


def test_errors_are_http_errors(client):
    assert client.post("/reset", json={"id": 10**9, "data_idx": 0}).status_code == 404
    env_id = new_env(client)
    assert client.post("/step", json={"id": env_id, "action": "look around"}).status_code == 409  # not reset
    assert client.post("/reset", json={"id": env_id, "data_idx": -1}).status_code == 400
    assert client.post("/reset", json={"id": env_id, "data_idx": 10**9}).status_code == 400
    assert client.post("/close", json={"id": env_id}).json() is True
    assert client.post("/close", json={"id": env_id}).status_code == 404
    assert client.post("/step", json={"id": env_id, "action": "look around"}).status_code == 404


def test_step_limit(server, client):
    old = server.config.step_limit
    server.config.step_limit = 3
    try:
        env_id = new_env(client)
        reset(client, env_id, encode("coin", "train", 0))
        assert not step(client, env_id, "inventory")["done"]
        assert not step(client, env_id, "inventory")["done"]
        p = step(client, env_id, "inventory")
        assert p["done"] and not p["success"] and p["num_moves"] == 3
    finally:
        server.config.step_limit = old


def test_success_reward_mode(server, client):
    server.config.reward_mode = "success"
    try:
        env_id = new_env(client)
        p = reset(client, env_id, encode("mapreader", "train", 0))
        gold = gold_path(server, env_id, p)
        partial = None
        for action in gold:
            if p["done"]:
                break
            p = step(client, env_id, action)
            if 0 < p["score"] < 1:
                partial = p
        assert partial is not None and partial["reward"] == 0.0  # partial credit is not rewarded
        assert p["reward"] == 1.0
    finally:
        server.config.reward_mode = "score"


def test_many_concurrent_episodes(client, server):
    """Like one AgentGym-RL rollout batch: 128 envs created, then stepped in parallel threads."""
    data = [encode(SUITES[i % 3].name, "train", i % 10) for i in range(128)]
    ids = [new_env(client) for _ in data]
    firsts = [reset(client, e, d) for e, d in zip(ids, data)]
    golds = [gold_path(server, e, p) for e, p in zip(ids, firsts)]

    def play(i):
        p = firsts[i]
        for action in golds[i]:
            if p["done"]:
                break
            p = step(client, ids[i], action)
        return p["reward"], p["done"]

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(play, range(len(ids))))
    assert all(r == (1.0, True) for r in results)
    for e in ids:
        assert client.post("/close", json={"id": e}).json() is True


@pytest.mark.parametrize("suite", [s.name for s in SUITES])
def test_concurrent_same_game_isolation(client, suite):
    """GRPO plays rollout.n copies of the same game at once; they must not interfere.

    Regression test for TextWorldExpress games with JVM-global state (sorting): play half
    of A, then create/reset/step another env on the same game and ask for a gold path
    (which builds a scratch game), then finish A.
    """
    for k in range(3):
        idx = encode(suite, "train", k)
        a = new_env(client)
        p = reset(client, a, idx)
        gold = client.get("/gold_path", params={"id": a}).json()
        half = len(gold) // 2
        for action in gold[:half]:
            p = step(client, a, action)
        b = new_env(client)
        pb = reset(client, b, idx)
        step(client, b, pb["valid_actions"][-1])
        client.get("/gold_path", params={"id": b})
        for action in gold[half:]:
            if p["done"]:
                break
            p = step(client, a, action)
        assert p["success"] and p["reward"] == 1.0, (suite, k, p["raw_observation"])
        client.post("/close", json={"id": a})
        client.post("/close", json={"id": b})


def test_isolated_games_get_and_release_a_dedicated_jvm(client, server):
    env_id = new_env(client)
    reset(client, env_id, encode("twc", "train", 0))
    assert server.envs[env_id].own_jvm is None
    reset(client, env_id, encode("sorting", "train", 0))
    jvm = server.envs[env_id].own_jvm
    assert jvm is not None and jvm.is_alive()
    client.post("/close", json={"id": env_id})
    jvm.proc.wait(timeout=30)
    assert not jvm.is_alive()


def test_create_close_does_not_leak(client, server):
    runtime = server._jvms[0].gateway.jvm.java.lang.Runtime.getRuntime()

    def used_mb():
        server._jvms[0].gateway.jvm.java.lang.System.gc()
        return (runtime.totalMemory() - runtime.freeMemory()) / 2**20

    def cycle(n):
        for i in range(n):
            env_id = new_env(client)
            reset(client, env_id, encode("twc", "train", i % 50))
            step(client, env_id, "look around")
            client.post("/close", json={"id": env_id})

    cycle(200)
    before = used_mb()
    cycle(1000)
    after = used_mb()
    assert len(server.envs) < 50
    assert after - before < 50, f"JVM heap grew from {before:.1f} MB to {after:.1f} MB"
