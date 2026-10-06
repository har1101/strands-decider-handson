"""Step 7: Strands Decider で「このツール呼び出しに人間の承認が要るか」を決める。

Strands の純正機能 ``HumanInTheLoop`` の ``classifier`` に、Decider で判定する関数を渡す:

- ``HumanInTheLoop``: ツール実行前に人間の承認を挟む intervention。``classifier`` が承認不要と判定したツールはそのまま実行する。
- ``classifier``: ``BeforeToolCallEvent`` を受け取り ``ClassifierResult`` を返す関数。ここで Decider に問い合わせる。
- ``ask="stdio"``: 承認をターミナルで y/n 入力で受け付ける。

``classifier=True`` にすると純正の LLM リスク判定が使われる。Decider に置き換えると、判定がローカルの 1 回の forward
pass で済む。

    uv run strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
    uv run python step7_tool_approval.py               # 組み込みの依頼を流す。承認を求められたら y/n で答える
    uv run python step7_tool_approval.py "README を読んで要約して"  # 任意の依頼を流す

ツールはすべて偽物(メモリ上の辞書を読み書きするだけ)。質問文・しきい値・方針はハンズオン用の例。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "strands-decider" / "examples" / "strands"))

from _client import Decider, ServerUnavailable  # noqa: E402
from strands import Agent, tool  # noqa: E402
from strands.hooks import BeforeToolCallEvent  # noqa: E402
from strands.vended_interventions.hitl import HumanInTheLoop  # noqa: E402
from strands.vended_interventions.hitl.classifier import ClassifierResult  # noqa: E402

MODEL_ID = "jp.anthropic.claude-haiku-4-5-20251001-v1:0"

SYSTEM_PROMPT = "You are an office assistant. Use the tools to complete the user's request, then report what you did."

QUESTIONS = {
    "needs_approval": Decider.noul(
        "Does this tool call change anything or send anything outside, so a human should approve it first?",
        {
            "true": (
                "it deletes, modifies or writes data, sends a message or email to anyone, moves money, "
                "or runs a state-changing command"
            ),
            "false": "it only reads, lists or searches information without changing anything or contacting anyone",
        },
    ),
}

# 承認が必要な確率がこれ以上なら人間に聞く。コード側が持つ方針のつまみ。
APPROVAL = 0.5

decider = Decider()


def decider_classifier(event: BeforeToolCallEvent, **kwargs) -> ClassifierResult:
    """ツール名と引数だけを Decider に見せて、人間の承認が要るかを判定する。"""
    tool_use = event.tool_use
    state = f"tool={tool_use['name']}\ninput={json.dumps(tool_use['input'], ensure_ascii=False)}"
    p = decider.ask(state, QUESTIONS)["needs_approval"]["noul"]
    verdict = "ask human" if p >= APPROVAL else "auto"
    print(f"  [decider] {tool_use['name']} p={p:.2f} -> {verdict} ({decider.last_latency_ms:.0f}ms)")
    return ClassifierResult(requires_human_in_the_loop=p >= APPROVAL, reason=f"Decider p={p:.2f}")


# ---- 偽物のツール ---------------------------------------------------------------

FILES = {
    "docs/README.md": "Strands Decider is a 2B decision model. It answers yes/no, choice and score questions.",
    "docs/CHANGELOG.md": "v19: added answer-adequacy training data.",
    "tmp/old.log": "debug log from last month",
}


@tool
def list_files(directory: str) -> list[str]:
    """List the files under a directory."""
    return [path for path in FILES if path.startswith(directory.rstrip("/") + "/")]


@tool
def read_file(path: str) -> str:
    """Read a text file."""
    return FILES.get(path, f"{path}: no such file")


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    print(f"  [tool] send_email to={to} subject={subject!r}")
    return f"sent to {to}"


@tool
def delete_file(path: str) -> str:
    """Delete a file."""
    FILES.pop(path, None)
    print(f"  [tool] delete_file {path}")
    return f"deleted {path}"


USER_REQUEST = (
    "docs フォルダのファイルを確認して README を要約し、alice@example.com にメールで送って。"
    "そのあと tmp/old.log を削除して。"
)


def main() -> int:
    try:
        print(decider.banner(), "\n")
    except ServerUnavailable as exc:
        print(exc)
        return 1

    agent = Agent(
        model=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
        tools=[list_files, read_file, send_email, delete_file],
        interventions=[HumanInTheLoop(classifier=decider_classifier, ask="stdio")],
        callback_handler=None,
    )
    message = " ".join(sys.argv[1:]) or USER_REQUEST
    print(f"USER: {message}\n")
    result = agent(message)
    print(f"\nASSISTANT: {str(result).strip()}\n")
    print(f"remaining files: {sorted(FILES)}")
    print(decider)
    return 0


if __name__ == "__main__":
    sys.exit(main())
