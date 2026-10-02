"""Step 5: Strands Decider で「どの LLM に答えさせるか」を決める(モデルルーティング)。

Strands の純正機能 ``ModelRouter`` に、Decider で判定する ``RoutingStrategy`` を渡す:

- ``ModelRouter``: 候補モデルの中から、invocation ごとに 1 つを選んで使うモデル。``Agent(model=...)`` に渡す。
- ``RoutingStrategy.select``: どの候補を使うかを決める非同期メソッド。ここで Decider に問い合わせる。

純正の ``ClassifierStrategy`` は、この判定自体を LLM に問い合わせる。Decider に置き換えると、判定がローカルの 1 回の
forward pass で済む。

    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
    python step5_model_router.py                       # 組み込みのサンプル依頼を順に流す
    python step5_model_router.py "この文を英訳して: 了解です"  # 任意の依頼を 1 件流す

質問文・選択肢・方針はハンズオン用の例。``ModelRouter`` は Strands 側で provisional(暫定)API とされている。
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "strands-decider" / "examples" / "strands"))

from _client import Decider, ServerUnavailable  # noqa: E402
from strands import Agent  # noqa: E402
from strands.models import BedrockModel  # noqa: E402
from strands.models.routing import ModelRouter, RoutingCandidate, RoutingContext  # noqa: E402

FAST_MODEL_ID = "jp.anthropic.claude-haiku-4-5-20251001-v1:0"
STRONG_MODEL_ID = "jp.anthropic.claude-sonnet-4-6"

SYSTEM_PROMPT = "You are a helpful assistant. Answer in the user's language, as concisely as the request allows."

QUESTIONS = {
    "model": Decider.choice(
        "Which model should handle this request?",
        {
            "fast": (
                "a small fast model is enough: greetings, short replies, translation of a phrase, "
                "simple lookups or formatting"
            ),
            "strong": (
                "needs a strong reasoning model: multi-step analysis, design, debugging, planning, "
                "long structured writing"
            ),
        },
    ),
}


def latest_user_text(context: RoutingContext) -> str:
    """直近のユーザーメッセージのテキストだけを取り出す。"""
    for message in reversed(context.messages):
        if message["role"] == "user":
            texts = [block["text"] for block in message["content"] if "text" in block]
            if texts:
                return "\n".join(texts)
    return ""


class DeciderStrategy:
    """Decider の choice で候補モデルを選ぶ RoutingStrategy。"""

    def __init__(self, decider: Decider) -> None:
        self._decider = decider
        self.last: dict | None = None

    async def select(self, context: RoutingContext, **kwargs) -> RoutingCandidate | None:
        # モデル呼び出しが失敗したあとの再選択では、別モデルへ切り替えずにエラーをそのまま返す。
        if context.attempts:
            return None
        # Decider クライアントは同期なので、イベントループを止めないよう別スレッドで呼ぶ。
        answer = (await asyncio.to_thread(self._decider.ask, latest_user_text(context), QUESTIONS))["model"]
        self.last = {
            "choice": answer["choice"],
            "confidence": round(answer["confidence"], 3),
            "decider_ms": self._decider.last_latency_ms,
        }
        return next(c for c in context.candidates if c.name == answer["choice"])


SAMPLE_REQUESTS = [
    "Translate 'good morning' into French.",
    "『承知しました』を丁寧なビジネスメールの一文に言い換えて",
    "Debug why async Python code deadlocks when two tasks acquire two locks in a different order, and propose a fix.",
    "新規事業の市場規模をフェルミ推定し、前提と感度分析も含めて説明して",
]


def main() -> int:
    decider = Decider()
    try:
        print(decider.banner(), "\n")
    except ServerUnavailable as exc:
        print(exc)
        return 1

    strategy = DeciderStrategy(decider)
    # 先頭の候補が既定値。strategy が None を返したときはこれが使われる。
    router = ModelRouter(
        [
            RoutingCandidate(BedrockModel(model_id=FAST_MODEL_ID), name="fast"),
            RoutingCandidate(BedrockModel(model_id=STRONG_MODEL_ID), name="strong"),
        ],
        strategy=strategy,
    )

    for message in sys.argv[1:] or SAMPLE_REQUESTS:
        agent = Agent(model=router, system_prompt=SYSTEM_PROMPT, callback_handler=None)
        started = time.perf_counter()
        result = agent(message)
        elapsed = time.perf_counter() - started
        usage = result.metrics.accumulated_usage
        model_id = FAST_MODEL_ID if strategy.last["choice"] == "fast" else STRONG_MODEL_ID

        print(f"USER: {message}")
        print(
            f"  [router] {strategy.last['choice']} -> {model_id} confidence={strategy.last['confidence']} "
            f"decider={strategy.last['decider_ms']:.0f}ms"
        )
        print(f"  [llm] total={elapsed:.1f}s input_tokens={usage['inputTokens']} output_tokens={usage['outputTokens']}")
        print(f"  ASSISTANT: {str(result).strip()[:300]}\n")

    print(decider)
    return 0


if __name__ == "__main__":
    sys.exit(main())
