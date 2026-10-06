"""Step 6: Strands Decider で「どのエージェントに渡すか」を決める(Graph の条件付きエッジ)。

Strands の純正機能 ``Graph`` で、受付 → 専門担当のワークフローを組む:

- ``GraphBuilder.add_node``: エージェントをノードとして登録する。
- ``GraphBuilder.add_edge(condition=...)``: 条件関数が True を返したエッジだけを辿る。ここで Decider に問い合わせる。

ワークフローの形はコードで固定し、分岐の判断だけを Decider に任せる。

    受付(intake) ──[Decider: billing?]──> 請求担当(billing)
                 ├─[Decider: tech?]─────> 技術担当(tech)
                 └─[Decider: sales?]────> 営業担当(sales)

    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
    python step6_graph_routing.py                          # 組み込みのサンプル問い合わせを順に流す
    python step6_graph_routing.py "ログインすると500エラーが出ます"  # 任意の問い合わせを 1 件流す

質問文・選択肢・方針はハンズオン用の例。
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "strands-decider" / "examples" / "strands"))

from _client import Decider, ServerUnavailable  # noqa: E402
from strands import Agent  # noqa: E402
from strands.multiagent import GraphBuilder  # noqa: E402
from strands.multiagent.graph import GraphState  # noqa: E402

MODEL_ID = "jp.anthropic.claude-haiku-4-5-20251001-v1:0"

INTAKE_PROMPT = (
    "あなたはサポートの受付担当です。お客様のメッセージを、社内チケット用に、"
    "問い合わせ内容を表す短い1文の日本語に書き直してください。その1文だけを出力してください。"
)

TEAMS = {
    "請求": "請求、請求書、支払い、返金",
    "技術": "バグ、エラー、クラッシュなど、製品を使うときの技術的な問題",
    "営業": "価格の質問、見積もり、割引、プランの購入やアップグレード",
}

TEAM_PROMPTS = {
    "請求": "あなたは SaaS 企業の請求担当チームです。",
    "技術": "あなたは SaaS 企業の技術サポートチームです。",
    "営業": "あなたは SaaS 企業の営業チームです。",
}

REPLY_RULES = (
    " お客様の元のメッセージに、お客様の言語で、3文以内で返信してください。"
    "対応に必要な情報があれば、お客様に尋ねてください。"
)

QUESTIONS = {
    "team": Decider.choice("このサポートチケットは、どのチームが対応すべきですか？", TEAMS),
}

decider = Decider()
decisions: dict[str, dict] = {}


@lru_cache(maxsize=128)
def route(ticket: str) -> str:
    """受付の要約文から担当チームを選ぶ。

    条件関数はエッジごとに呼ばれる(3 本なら 3 回)ので、同じ要約文の判定はキャッシュして Decider を 1 回で済ませる。
    """
    answer = decider.ask(f"サポートチケット: {ticket}", QUESTIONS)["team"]
    decisions[ticket] = {
        "team": answer["choice"],
        "confidence": round(answer["confidence"], 3),
        "decider_ms": decider.last_latency_ms,
    }
    return answer["choice"]


def routed_to(team: str):
    """受付ノードの出力を Decider が ``team`` に振り分けたときだけ True を返す条件関数を作る。"""

    def condition(state: GraphState) -> bool:
        return route(str(state.results["受付"].result).strip()) == team

    return condition


def build_graph():
    builder = GraphBuilder()
    builder.add_node(Agent(model=MODEL_ID, system_prompt=INTAKE_PROMPT, callback_handler=None), "受付")
    for team in TEAMS:
        agent = Agent(model=MODEL_ID, system_prompt=TEAM_PROMPTS[team] + REPLY_RULES, callback_handler=None)
        builder.add_node(agent, team)
        builder.add_edge("受付", team, condition=routed_to(team))
    builder.set_entry_point("受付")
    builder.set_max_node_executions(2)  # 受付 1 回 + 担当 1 回
    return builder.build()


SAMPLE_REQUESTS = [
    "今月、クレジットカードに二重で請求されています。",
    "ログインすると500エラーが出ます",
    "エンタープライズプランを200席で使いたいので見積もりが欲しいです",
]


def main() -> int:
    try:
        print(decider.banner(), "\n")
    except ServerUnavailable as exc:
        print(exc)
        return 1

    for message in sys.argv[1:] or SAMPLE_REQUESTS:
        result = build_graph()(message)
        ticket = str(result.results["受付"].result).strip()
        decision = decisions[ticket]
        path = " -> ".join(node.node_id for node in result.execution_order)

        print(f"USER: {message}")
        print(f"  [受付] {ticket}")
        print(
            f"  [route] {decision['team']} confidence={decision['confidence']} "
            f"decider={decision['decider_ms']:.0f}ms path={path}"
        )
        print(f"  [{decision['team']}] {str(result.results[decision['team']].result).strip()}\n")

    print(decider)
    return 0


if __name__ == "__main__":
    sys.exit(main())
