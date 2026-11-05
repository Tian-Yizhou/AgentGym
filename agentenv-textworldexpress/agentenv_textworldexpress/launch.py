"""
Entrypoint for the TextWorldExpress agent environment.
"""

import argparse
import os

import uvicorn


def launch():
    """entrypoint for `textworldexpress` command"""

    parser = argparse.ArgumentParser(description="TextWorldExpress environment server for AgentGym.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--reward-mode",
        choices=["score", "success"],
        default=os.environ.get("TWX_REWARD_MODE", "score"),
        help="'score': TextWorldExpress normalized score in [0, 1] (partial credit). "
        "'success': 1.0 if the task was completed, else 0.0.",
    )
    parser.add_argument(
        "--show-valid-actions",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("TWX_SHOW_VALID_ACTIONS", "1").lower() in {"1", "true", "yes", "on"},
        help="Append the list of currently valid actions to every observation (default: on).",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=int(os.environ.get("TWX_STEP_LIMIT", 100)),
        help="Episode ends after this many steps (default 100, TextWorldExpress's default). "
        "Keep it >= the RL interaction rounds so the trainer controls the horizon.",
    )
    parser.add_argument("--num-jvms", type=int, default=int(os.environ.get("TWX_NUM_JVMS", 1)))
    parser.add_argument("--jvm-heap", type=str, default=os.environ.get("TWX_JVM_HEAP", "2g"))
    args = parser.parse_args()

    # The app reads its configuration from the environment at startup (see ServerConfig.from_env).
    os.environ["TWX_REWARD_MODE"] = args.reward_mode
    os.environ["TWX_SHOW_VALID_ACTIONS"] = "1" if args.show_valid_actions else "0"
    os.environ["TWX_STEP_LIMIT"] = str(args.step_limit)
    os.environ["TWX_NUM_JVMS"] = str(args.num_jvms)
    os.environ["TWX_JVM_HEAP"] = args.jvm_heap

    # Single worker on purpose: environment ids live in this process's memory.
    uvicorn.run("agentenv_textworldexpress:app", host=args.host, port=args.port)
