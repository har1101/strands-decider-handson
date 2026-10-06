"""Step 7: Strands Decider で「このツール呼び出しに人間の承認が要るか」を決める。

Strands の純正機能 ``HumanInTheLoop`` の ``classifier`` に、Decider で判定する関数を渡す:

- ``HumanInTheLoop``: ツール実行前に人間の承認を挟む intervention。``classifier`` が承認不要と判定したツールはそのまま実行する。
- ``classifier``: ``BeforeToolCallEvent`` を受け取り ``ClassifierResult`` を返す関数。ここで Decider に問い合わせる。
- ``ask="stdio"``: 承認をターミナルで y/n 入力で受け付ける。

``classifier=True`` にすると純正の LLM リスク判定が使われる。Decider に置き換えると、判定がローカルの 1 回の forward
pass で済む。

    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
    python step7_tool_approval.py               # 組み込みの依頼を流す。承認を求められたら y/n で答える
    python step7_tool_approval.py "README を読んで要約して"  # 任意の依頼を流す

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

SYSTEM_PROMPT = "あなたは事務アシスタントです。ツールを使ってユーザーの依頼を完了し、何をしたかを報告してください。"

QUESTIONS = {
    "needs_approval": Decider.noul(
        "このツール呼び出しは、何かを変更する、または外部に何かを送るため、先に人間が承認すべきですか？",
        {
            "true": (
                "データの削除・変更・書き込み、誰かへのメッセージやメールの送信、送金、"
                "状態を変えるコマンドの実行を行う"
            ),
            "false": "情報の読み取り、一覧表示、検索だけを行い、何も変更せず、誰にも連絡しない",
        },
    ),
}

# 承認が必要な確率がこれ以上なら人間に聞く。コード側が持つ方針のつまみ。
APPROVAL = 0.5

decider = Decider()


def decider_classifier(event: BeforeToolCallEvent, **kwargs) -> ClassifierResult:
    """ツール名と引数だけを Decider に見せて、人間の承認が要るかを判定する。"""
    tool_use = event.tool_use
    state = f"ツール名: {tool_use['name']}\n入力: {json.dumps(tool_use['input'], ensure_ascii=False)}"
    p = decider.ask(state, QUESTIONS)["needs_approval"]["noul"]
    verdict = "ask human" if p >= APPROVAL else "auto"
    print(f"  [decider] {tool_use['name']} p={p:.2f} -> {verdict} ({decider.last_latency_ms:.0f}ms)")
    return ClassifierResult(requires_human_in_the_loop=p >= APPROVAL, reason=f"Decider p={p:.2f}")


# ---- 偽物のツール ---------------------------------------------------------------

FILES = {
    "docs/README.md": "Strands Decider は 2B の判定モデルです。yes/no、選択、スコアの質問に答えます。",
    "docs/CHANGELOG.md": "v19: 回答の妥当性に関する学習データを追加。",
    "tmp/old.log": "先月のデバッグログ",
}


@tool
def list_files(directory: str) -> list[str]:
    """ディレクトリの下にあるファイルを一覧表示する。"""
    return [path for path in FILES if path.startswith(directory.rstrip("/") + "/")]


@tool
def read_file(path: str) -> str:
    """テキストファイルを読む。"""
    return FILES.get(path, f"{path}: ファイルが見つかりません")


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """メールを送る。"""
    print(f"  [tool] send_email to={to} subject={subject!r}")
    return f"{to} に送信しました"


@tool
def delete_file(path: str) -> str:
    """ファイルを削除する。"""
    FILES.pop(path, None)
    print(f"  [tool] delete_file {path}")
    return f"{path} を削除しました"


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
