"""Step 4: Strands Decider で「LLM で回答してよいか / 人間にエスカレーションすべきか」を振り分ける。

Strands の純正機能だけで分岐させる:

- ``InterventionHandler.before_invocation``: エージェントがリクエストを処理し始める前(LLM 呼び出し前)に走るフック。
- ``Proceed``: そのまま LLM(Bedrock)が回答する。
- ``Deny``: invocation をキャンセルする。LLM は一度も呼ばれず、``reason`` がアシスタントの返答になる。
- ``agent.state``: 判定結果を呼び出し側に渡す。

    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
    python step4_escalation_gate.py                      # 組み込みのサンプル問い合わせを順に流す
    python step4_escalation_gate.py "返金してください"    # 任意の問い合わせを 1 件流す

質問文・しきい値・方針はハンズオン用の例。実トラフィックに合わせて調整すること。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "strands-decider" / "examples" / "strands"))

from _client import Decider, ServerUnavailable  # noqa: E402
from strands import Agent  # noqa: E402
from strands.hooks import BeforeInvocationEvent  # noqa: E402
from strands.interventions import Deny, InterventionHandler, Proceed  # noqa: E402

MODEL_ID = "jp.anthropic.claude-haiku-4-5-20251001-v1:0"

SYSTEM_PROMPT = (
    "あなたは SaaS 製品のカスタマーサポート担当アシスタントです。"
    "機能、プラン、使い方に関する一般的な質問に、簡潔に答えてください。"
)

QUESTIONS = {
    "route": Decider.choice(
        "このサポートへの問い合わせは、誰が対応すべきですか？",
        {
            "AIアシスタント": (
                "製品ドキュメントをもとに、AIアシスタントだけで十分に答えられる一般的な質問。"
                "使い方、機能、プラン、トラブルシューティングの手順"
            ),
            "担当者": (
                "人間のスタッフが必要な問い合わせ。返金、請求のトラブル、補償、"
                "アカウントのセキュリティ事故、法的な脅し、お客様自身のアカウントやお金に関わる操作"
            ),
        },
    ),
}

# 「担当者」の確率がこれ以上なら人間へ。コード側が持つ方針のつまみ。
HUMAN = 0.7

HANDOFF_MESSAGE = "担当者におつなぎします。内容を確認のうえ、担当者からご連絡いたします。"


class EscalationGate(InterventionHandler):
    """LLM を呼ぶ前に、Decider で問い合わせの担当(AI / 人間)を判定する。"""

    name = "decider-escalation-gate"

    def __init__(self, decider: Decider) -> None:
        self._decider = decider

    def before_invocation(self, event: BeforeInvocationEvent, **kwargs):
        text = "\n".join(
            block["text"]
            for message in event.messages or []
            if message["role"] == "user"
            for block in message["content"]
            if "text" in block
        )
        answer = self._decider.ask(f"お客様がサポートに次のメッセージを送りました:\n{text}", QUESTIONS)["route"]
        p_human = answer["probabilities"]["担当者"]
        route = "human" if p_human >= HUMAN else "llm"
        event.agent.state.set(
            "escalation",
            {
                "route": route,
                "p_human": round(p_human, 3),
                "confidence": round(answer["confidence"], 3),
                "decider_ms": self._decider.last_latency_ms,
            },
        )
        if route == "human":
            return Deny(reason=HANDOFF_MESSAGE)
        return Proceed()


def escalate_to_human(message: str, decision: dict) -> None:
    """人間側の分岐。実運用ではチケット起票や Slack 通知などに置き換える。"""
    print(f"  [human] チケット起票: p_human={decision['p_human']} message={message!r}")


SAMPLE_REQUESTS = [
    "データをCSVで書き出すにはどうすればいいですか？",
    "パスワードを忘れました。どうすればいいですか？",
    "注文番号1234で二重に請求されています。すぐに返金してください。",
    "不正アクセスされました。身に覚えのない国からログインされ、メールアドレスを勝手に変更されています。",
    "二重に請求されています。返金してください。",
]


def handle(decider: Decider, message: str) -> None:
    agent = Agent(
        model=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
        interventions=[EscalationGate(decider)],
        callback_handler=None,
    )
    result = agent(message)
    decision = agent.state.get("escalation")
    llm_tokens = result.metrics.accumulated_usage["inputTokens"]

    print(f"USER: {message}")
    print(
        f"  [gate] route={decision['route']} p_human={decision['p_human']} "
        f"confidence={decision['confidence']} decider={decision['decider_ms']:.0f}ms "
        f"llm_input_tokens={llm_tokens}"
    )
    if decision["route"] == "human":
        escalate_to_human(message, decision)
        print(f"  ASSISTANT: {HANDOFF_MESSAGE}\n")
    else:
        print(f"  [llm] ASSISTANT: {str(result).strip()}\n")


def main() -> int:
    decider = Decider()
    try:
        print(decider.banner(), "\n")
    except ServerUnavailable as exc:
        print(exc)
        return 1
    for message in sys.argv[1:] or SAMPLE_REQUESTS:
        handle(decider, message)
    print(decider)
    return 0


if __name__ == "__main__":
    sys.exit(main())
