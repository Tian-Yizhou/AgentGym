import re
from typing import Any, Mapping

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class TextWorldExpressEnvClient(BaseEnvClient):
    """Client for the TextWorldExpress env server (``AgentGym/agentenv-textworldexpress``).

    The server covers all TextWorldExpress games (cookingworld, coin, twc, mapreader,
    arithmetic, sorting, simonsays, peckingorder); ``data_idx`` selects game, fold and seed
    (see ``agentenv_textworldexpress/games.py``). The game-specific goal is part of the first
    observation, so the conversation start below is generic.

    ``StepOutput.reward`` is the episode's current cumulative reward, which is what AgentGym-RL
    expects from the last step of a trajectory: TextWorldExpress's normalized score (1.0 =
    solved, partial credit in between, negative when an arithmetic/sorting/simonsays/
    peckingorder episode fails), or 1/0 success if the server runs with ``--reward-mode success``.
    """

    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": (
                    "You are an agent playing a text-based game. Every round I will give you an observation "
                    "of the environment, followed by the list of actions that are currently available. "
                    "Your goal is to complete the task described in the first observation, using as few "
                    "actions as possible.\n\n"
                    "Your response should use the following format:\n"
                    "Thought:\n<your thoughts>\n\n"
                    "Action:\n<exactly one action, copied verbatim from the available actions>"
                ),
            }
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "OK. I'll follow your instructions and try my best to solve the task.",
            }
        ),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base.rstrip("/")
        self.timeout = timeout
        self.data_len = data_len
        self.info: dict[str, Any] = {}

        res = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        self._raise_for_status(res, "create")
        self.env_id = res.json()["id"]

    def __len__(self):
        return self.data_len

    @staticmethod
    def _raise_for_status(res: requests.Response, what: str) -> None:
        if res.status_code != 200:
            try:
                detail = res.json().get("detail", res.text)
            except ValueError:
                detail = res.text
            raise RequestException(f"TextWorldExpress /{what} failed ({res.status_code}): {detail}")

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(f"{self.env_server_base}/{path}", json=data, timeout=self.timeout)
        self._raise_for_status(res, path)
        return res.json()

    def _get(self, path: str) -> Any:
        res = requests.get(f"{self.env_server_base}/{path}", params={"id": self.env_id}, timeout=self.timeout)
        self._raise_for_status(res, path)
        return res.json()

    @staticmethod
    def parse_action(text: str) -> tuple[str | None, str | None]:
        """Extract the action from an LLM response. Returns ``(action, error_message)``."""
        matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", text, re.DOTALL)
        if len(matches) > 1:
            return None, "Error: Only one 'Action' is allowed per response. Please adjust your response."
        if not matches or not matches[0].strip():
            return None, (
                "Error: No action found. End your response with 'Action:' followed by exactly one of "
                "the available actions."
            )
        return matches[0].strip(), None

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        parsed, error = self.parse_action(action)
        if error is not None:
            # A format error does not advance the game and keeps the score earned so far.
            return StepOutput(state=error, reward=self.info.get("reward", 0.0), done=False)
        response = self._post("step", {"action": parsed})
        self.info = response
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, data_idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": data_idx})
        self.info = response
        return response

    def close(self):
        return self._post("close", {})


class TextWorldExpressTask(BaseTask):
    env_client_cls = TextWorldExpressEnvClient
    env_name = "TextWorldExpress"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
