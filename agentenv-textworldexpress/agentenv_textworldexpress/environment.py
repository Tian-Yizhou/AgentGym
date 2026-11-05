"""Server-side manager for TextWorldExpress environments.

Design notes
------------
* The stock ``textworld_express.TextWorldExpressEnv`` starts one JVM per instance.
  AgentGym-RL creates one env per rollout sample per training step (batch x n, e.g.
  16 x 8 = 128), so that would mean hundreds of short-lived Java processes. Instead,
  a small pool of JVMs is started once, and each env is a separate
  ``textworldexpress.runtime.PythonInterface`` object inside one of them. All game
  state lives in that object, so environments sharing a JVM are independent.
* Exception: games in ``ISOLATED_GAMES`` keep JVM-global mutable state, so two games of
  that kind in one JVM corrupt each other (in ``sorting``, ``SortingProblemGenerator`` is a
  Scala singleton whose item objects are reused by every game built from the same problem;
  generating a second game moves those items into the new game). Envs playing such a game
  get a dedicated JVM, which is what the stock wrapper does for every env. This costs about
  0.7 s of JVM startup on the reset that first needs it.
  ``tests/test_server.py::test_concurrent_same_game_isolation`` checks every suite for this.
* Episode bookkeeping (score, done, step limit) mirrors
  ``TextWorldExpressEnv.step`` from TextWorldExpress 1.1.0.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import shutil
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import orjson
from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway
from py4j.protocol import Py4JError
from textworld_express.constants import BASEPATH, JAR_PATH

from .games import SUITES, GameSpec, Suite, decode

logger = logging.getLogger(__name__)

REWARD_MODES = ("score", "success")

# Games that must not share a JVM with another game of the same kind (see module docstring).
ISOLATED_GAMES = frozenset({"sorting"})
ISOLATED_JVM_HEAP = "512m"


class EnvError(Exception):
    """An error with an HTTP status code, turned into a JSON error response by the server."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ServerConfig:
    # "score": TWX normalized score (1.0 = solved; partial credit in between; negative when an
    #          episode fails in arithmetic/sorting/simonsays/peckingorder) -- TWX's own definition.
    # "success": 1.0 iff the task was solved, else 0.0.
    reward_mode: str = "score"
    show_valid_actions: bool = True  # append the list of valid actions to every observation
    step_limit: int = 100  # episode ends after this many steps (TWX's default envStepLimit)
    num_jvms: int = 1  # number of Java processes in the pool
    jvm_heap: str = "2g"  # -Xmx for each Java process

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Read the configuration from ``TWX_*`` environment variables (set by ``launch.py``)."""
        cfg = cls(
            reward_mode=os.environ.get("TWX_REWARD_MODE", cls.reward_mode),
            show_valid_actions=_env_bool("TWX_SHOW_VALID_ACTIONS", cls.show_valid_actions),
            step_limit=int(os.environ.get("TWX_STEP_LIMIT", cls.step_limit)),
            num_jvms=int(os.environ.get("TWX_NUM_JVMS", cls.num_jvms)),
            jvm_heap=os.environ.get("TWX_JVM_HEAP", cls.jvm_heap),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {self.reward_mode!r}")
        if self.step_limit < 1:
            raise ValueError(f"step_limit must be >= 1, got {self.step_limit}")
        if self.num_jvms < 1:
            raise ValueError(f"num_jvms must be >= 1, got {self.num_jvms}")


class _JVM:
    """One Java process that hosts many ``PythonInterface`` game objects."""

    def __init__(self, heap: str):
        self.heap = heap
        self.lock = threading.Lock()
        self._start()

    def _start(self) -> None:
        port, proc = launch_gateway(
            classpath=JAR_PATH,
            cwd=BASEPATH,  # TextWorldExpress loads its data files relative to the package dir
            javaopts=[f"-Xmx{self.heap}"],
            die_on_exit=True,  # the JVM exits when this Python process exits
            redirect_stderr=sys.__stderr__,  # surface JVM crashes (e.g. OutOfMemoryError) in the server log
            return_proc=True,
        )
        self.proc = proc
        self.gateway = JavaGateway(
            gateway_parameters=GatewayParameters(port=port, auto_field=True, auto_close=True),
            java_process=proc,
        )
        logger.info("Started TextWorldExpress JVM (pid=%s, port=%s, heap=%s)", proc.pid, port, self.heap)

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def new_interface(self):
        with self.lock:
            if not self.is_alive():
                logger.warning("TextWorldExpress JVM (pid=%s) died; restarting it", self.proc.pid)
                self._start()
            gateway = self.gateway
        return gateway.jvm.textworldexpress.runtime.PythonInterface()

    def shutdown(self) -> None:
        try:
            self.gateway.shutdown()
        except Exception:  # noqa: BLE001 - best effort on teardown
            pass
        if self.is_alive():
            self.proc.kill()


@dataclass
class _Episode:
    interface: Any
    own_jvm: Optional[_JVM] = None  # dedicated JVM, only for envs that played an ISOLATED_GAMES game
    lock: threading.Lock = field(default_factory=threading.Lock)
    loaded_suite: Optional[Suite] = None
    spec: Optional[GameSpec] = None
    task_description: str = ""
    last: dict = field(default_factory=dict)  # last parsed TWX step result
    num_moves: int = 0
    done: bool = False
    payload: dict = field(default_factory=dict)  # last response sent to the client


def _is_twx_error(result: dict) -> bool:
    """True for TextWorldExpress's internal error results (``StepResult.mkErrorMessage``).

    Not to be confused with in-game failures: arithmetic, sorting, simonsays and peckingorder
    legitimately end a failed episode with a negative score (and ``taskfailure``).
    """
    return not result.get("validActions") and str(result.get("observation", "")).startswith("ERROR")


def normalize_action(action: str) -> str:
    """Light, format-only cleanup of an action string (no fuzzy matching)."""
    action = action.strip().strip("`\"'").strip()
    action = action.rstrip(".").strip()
    return " ".join(action.lower().split())


class TextWorldExpressServer:
    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.config.validate()
        if shutil.which("java") is None:
            raise RuntimeError(
                "Java (>= 1.8) was not found on PATH. TextWorldExpress needs it, e.g. "
                "`conda install -c conda-forge openjdk=17` in this environment."
            )
        self._jvms = [_JVM(self.config.jvm_heap) for _ in range(self.config.num_jvms)]
        self._round_robin = itertools.cycle(range(len(self._jvms)))
        self._lock = threading.Lock()
        self._next_id = 0
        self.envs: dict[int, _Episode] = {}
        self._validate_suites()

    # ------------------------------------------------------------------ lifecycle

    def _validate_suites(self) -> None:
        """Fail fast on a bad registry entry instead of on the first reset during training."""
        iface = self._jvms[0].new_interface()
        for suite in SUITES:
            msg = iface.load(suite.game, suite.params)
            if msg:
                raise ValueError(f"Invalid TextWorldExpress suite {suite}: {msg}")
            result = orjson.loads(iface.generateNewGameJSON(0, "train", False))
            if _is_twx_error(result) or not result.get("validActions"):
                raise ValueError(f"TextWorldExpress suite {suite} failed to generate a game: {result}")

    def shutdown(self) -> None:
        for ep in self.envs.values():
            if ep.own_jvm is not None:
                ep.own_jvm.shutdown()
        self.envs.clear()
        for jvm in self._jvms:
            jvm.shutdown()

    # ------------------------------------------------------------------ helpers

    def _get(self, env_id: int) -> _Episode:
        try:
            return self.envs[env_id]
        except KeyError:
            raise EnvError(404, f"Environment {env_id} does not exist (never created, or already closed).") from None

    def _reward(self, score: float, success: bool) -> float:
        if self.config.reward_mode == "success":
            return 1.0 if success else 0.0
        return float(score)

    def _format(self, observation: str, valid_actions: list[str], *, task: str = "", done: bool = False) -> str:
        parts = []
        if task:
            parts.append(task.strip())
        parts.append(observation.strip())
        if self.config.show_valid_actions and not done:
            parts.append("Available actions: " + json.dumps(valid_actions))
        return "\n\n".join(parts)

    def _build_payload(self, ep: _Episode, observation: str) -> dict:
        res = ep.last
        valid_actions = sorted(res["validActions"])
        success, failure = bool(res["tasksuccess"]), bool(res["taskfailure"])
        score = float(res["score"])
        return {
            "observation": observation,
            "reward": self._reward(score, success),
            "score": score,
            "done": ep.done,
            "success": success,
            "failure": failure,
            "num_moves": ep.num_moves,
            "valid_actions": valid_actions,
            "raw_observation": res["observation"],
            "task_description": ep.task_description,
            "data_idx": ep.spec.data_idx,
            "suite": ep.spec.suite.name,
            "game": ep.spec.suite.game,
            "fold": ep.spec.fold,
            "seed": ep.spec.seed,
        }

    # ------------------------------------------------------------------ API

    def create(self) -> dict:
        with self._lock:
            env_id = self._next_id
            self._next_id += 1
            jvm = self._jvms[next(self._round_robin)]
        try:
            interface = jvm.new_interface()
        except Py4JError as e:
            raise EnvError(503, f"Failed to create a TextWorldExpress game: {e}") from e
        self.envs[env_id] = _Episode(interface=interface)
        return {"id": env_id}

    def reset(self, env_id: int, data_idx: int) -> dict:
        ep = self._get(env_id)
        try:
            spec = decode(data_idx)
        except ValueError as e:
            raise EnvError(400, str(e)) from e
        with ep.lock:
            try:
                if spec.suite.game in ISOLATED_GAMES and ep.own_jvm is None:
                    ep.own_jvm = _JVM(ISOLATED_JVM_HEAP)
                    ep.interface = ep.own_jvm.new_interface()
                    ep.loaded_suite = None
                if ep.loaded_suite != spec.suite:
                    msg = ep.interface.load(spec.suite.game, spec.suite.params)
                    if msg:
                        raise EnvError(500, f"TextWorldExpress failed to load {spec.suite}: {msg}")
                    ep.loaded_suite = spec.suite
                result = orjson.loads(ep.interface.generateNewGameJSON(spec.seed, spec.fold, False))
                task_description = ep.interface.getTaskDescription()
            except Py4JError as e:
                raise EnvError(503, f"TextWorldExpress JVM error during reset: {e}") from e
            if _is_twx_error(result):
                raise EnvError(500, f"TextWorldExpress failed to generate game {spec}: {result['observation']}")
            ep.spec, ep.last, ep.task_description = spec, result, task_description
            ep.num_moves, ep.done = 0, False
            observation = self._format(result["observation"], sorted(result["validActions"]), task=task_description)
            ep.payload = self._build_payload(ep, observation)
            return ep.payload

    def step(self, env_id: int, action: str) -> dict:
        ep = self._get(env_id)
        with ep.lock:
            if ep.spec is None:
                raise EnvError(409, f"Environment {env_id} has not been reset yet.")
            if ep.done:
                raise EnvError(409, f"The episode in environment {env_id} has finished; call /reset first.")
            normalized = normalize_action(action)
            canonical = {a.lower(): a for a in ep.last["validActions"]}
            twx_action = canonical.get(normalized, normalized)
            try:
                result = orjson.loads(ep.interface.stepJSON(twx_action))
            except Py4JError as e:
                raise EnvError(503, f"TextWorldExpress JVM error during step: {e}") from e
            if _is_twx_error(result):
                raise EnvError(500, f"TextWorldExpress step failed: {result['observation']}")
            ep.last = result
            ep.num_moves += 1
            ep.done = (
                result["score"] >= 1.0
                or bool(result["tasksuccess"])
                or bool(result["taskfailure"])
                or ep.num_moves >= self.config.step_limit
            )
            observation = self._format(result["observation"], sorted(result["validActions"]), done=ep.done)
            ep.payload = self._build_payload(ep, observation)
            ep.payload["action"] = twx_action
            ep.payload["valid_action"] = twx_action in canonical.values()
            return ep.payload

    def observation(self, env_id: int) -> str:
        ep = self._get(env_id)
        if not ep.payload:
            raise EnvError(409, f"Environment {env_id} has not been reset yet.")
        return ep.payload["observation"]

    def detail(self, env_id: int) -> dict:
        ep = self._get(env_id)
        if not ep.payload:
            raise EnvError(409, f"Environment {env_id} has not been reset yet.")
        return ep.payload

    def gold_path(self, env_id: int) -> list:
        """TextWorldExpress's reference solution for the env's current game (for tests/debugging/SFT).

        Computed on a scratch game object, so the live episode is not touched. TWX gold agents
        random-walk: paths solve the task but are not shortest, and differ between calls.
        """
        ep = self._get(env_id)
        if ep.spec is None:
            raise EnvError(409, f"Environment {env_id} has not been reset yet.")
        spec = ep.spec
        # A scratch game of an isolated kind would corrupt live games in the same JVM.
        temp_jvm = _JVM(ISOLATED_JVM_HEAP) if spec.suite.game in ISOLATED_GAMES else None
        try:
            scratch = (temp_jvm or self._jvms[0]).new_interface()
            msg = scratch.load(spec.suite.game, spec.suite.params)
            if msg:
                raise EnvError(500, f"TextWorldExpress failed to load {spec.suite}: {msg}")
            scratch.generateNewGameJSON(spec.seed, spec.fold, True)
            return list(scratch.getGoldActionSequence())
        except Py4JError as e:
            raise EnvError(503, f"TextWorldExpress JVM error while computing the gold path: {e}") from e
        finally:
            if temp_jvm is not None:
                temp_jvm.shutdown()

    def close(self, env_id: int) -> bool:
        # Dropping the last Python reference lets py4j release the Java object.
        ep = self.envs.pop(env_id, None)
        if ep is None:
            raise EnvError(404, f"Environment {env_id} does not exist (never created, or already closed).")
        if ep.own_jvm is not None:
            ep.own_jvm.shutdown()
        return True
