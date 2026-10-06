# Strands Decider 2B ハンズオン

Strands Decider 2B を CLI → HTTP サーバー → Strands エージェント組み込みの順に試す手順。

- ブログ: https://strandsagents.com/blog/introducing-strands-decider/
- GitHub: https://github.com/strands-labs/strands-decider
- 重み: https://huggingface.co/StrandsAgents

## Strands Decider とは

- テキストを生成せず、**選択肢から選ぶ / スケールで採点する**だけの「decision model(system one model)」。
- 質問タイプは 3 種類。
  - `noul`: Yes/No を 0〜1 の確率で返す(1 に近いほど Yes)
  - `choice`: N 個の選択肢から 1 つ選ぶ
  - `score`: 順序付きルーブリックで採点する
- すべての回答に calibrated confidence が付く。未知の短い分類タスクでは、confidence 0.9 以上の回答の約 95% が正解。
- 構成: Qwen3.5-2B の LM head を外し、約 100 万パラメータの pointer head に置き換えたもの。torso は rank-16 LoRA で fine-tune。1 回の forward pass で答え、生成ループは持たない。
- 速度: RTX 3090 で中央値 115ms/問。CPU と Apple silicon でも推論できる。
- 同じ state への複数の質問は、state を 1 回読むだけで済むので低コスト。
- 想定用途: モデルルーティング、ツール選択、ツール引数チェック、トリアージ、ガードレール、評価、ハイブリッドエージェント(難しい判断は LLM、定型判断は Decider)。
- 苦手なこと: 複雑な推論、コーディング、チャット、要約など、テキスト生成が必要なタスク。

## 前提環境

| 項目 | 要件 | 備考 |
| --- | --- | --- |
| Python | 3.10 以上 | `pyproject.toml` の `requires-python = ">=3.10"` |
| ディスク | 10GB 程度の空き | 1.9B パラメータの重みは bf16 で約 3.8GB。これに torch などが加わる |
| メモリ | 8GB 以上を推奨 | fp32 で読み込むと重みだけで約 7.6GB |
| GPU | 不要 | CPU でも推論できる。レイテンシは GPU より遅くなる |
| AWS 認証情報 | Step 3 のみ必要 | Amazon Bedrock を呼べること |

### Lambda MicroVM 環境の場合

デフォルトのベースライン(2GB / 1vCPU)では、ディスク上限が 8GB、ピーク時メモリが 8GB で足りない。ディスク上限はベースラインのメモリ量で決まるので、MicroVM Image の `MinimumMemoryInMiB` を `4096` 以上に上げ、**新しい MicroVM を起動する**。

| ベースライン | ピーク | ディスク上限 |
| --- | --- | --- |
| 2GB / 1vCPU | 8GB / 4vCPU | 8GB |
| 4GB / 2vCPU | 16GB / 8vCPU | 16GB |
| 8GB / 4vCPU | 32GB / 16vCPU | 32GB |

出典: https://docs.aws.amazon.com/lambda/latest/dg/microvms-images.html#microvms-images-sizing

反映されたかは次のコマンドで確認する。

```bash
free -g; nproc; df -h /
```

#### 推論が止まる場合(NVPL BLAS の回避策)

MicroVM では、`Loading weights: 100%` のあと推論が終わらず、CPU 1 コアが 100% のまま止まることがある。原因はメモリ不足ではない。行列演算ライブラリが無限ループしている。

- aarch64 向けの PyTorch(例: `2.14.1+cu130`)は、行列演算に NVIDIA の NVPL BLAS を同梱している。
- MicroVM の CPU は Neoverse V1(Graviton3、MIDR `0xd40`)だが、VM 側で SVE が無効になっている(`/proc/cpuinfo` の Flags に `sve` がない)。
- NVPL は CPU 名から Neoverse V1 向けの処理を選ぶ。`NVPL_BLAS_VERBOSE=1` で実行すると `Platform: Neoverse V1, cores:8 sve_width:0` と表示され、SVE の幅が 0 のまま動いていることがわかる。この状態では `torch.rand(32,1) @ torch.rand(1,20)` のような小さな行列積でも `sgemm` から戻らない。
- `OMP_NUM_THREADS=1` などのスレッド数の指定では直らない。

NVPL に SVE を使わない汎用の CPU タイプ(`1` = `Generic ASIMD`)を使わせると回避できる。`strands-decider` を実行するシェル(Step 2 の `serve` を起動するシェルも含む)で、事前に次を設定する。

```bash
export NVPL_BLAS_DEBUG_CPU_TYPE=1
```

試した値のうち、`generic`、`cortexa57`、`neoverse_n1`、`2` は同じように止まり、`0` は segfault した。動いたのは `1` だけ。

影響を受けるかどうかは、次のコマンドで確認できる。10 秒以内に `ok` が出なければ影響を受けている。

```bash
timeout 10 python -c "import torch; print('ok', (torch.rand(32,1) @ torch.rand(1,20)).shape)"
```

### Python 環境の作成

```bash
uv venv -p 3.14
source .venv/bin/activate
uv pip install strands-decider
```

## Step 1: CLI で 3 種類の質問を試す

AWS は不要。初回実行時に Hugging Face からモデルがダウンロードされる。

### choice(選択)

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "助けて！請求金額が3日前からずっと間違っています！" \
  --choice "どのチームが対応すべきですか？=請求,営業,技術"
```

出力例:

```text
choice_0 -> 請求 (confidence 0.398)
    請求                       0.599
    技術                       0.265
    営業                       0.136
```

### noul(Yes/No)

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "助けて！請求金額が3日前からずっと間違っています！" \
  --noul "緊急性を感じさせますか？"
```

```text
noul_0 noul = 0.789
```

### score(採点)

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "助けて！請求金額が3日前からずっと間違っています！" \
  --score "書き手はどれくらい苛立っていますか？=落ち着いている,苛立っている,落ち込んでいる"
```

```text
score_0 score = 1.17 (confidence 0.516)
    0: 落ち着いている                                  0.139
    1: 苛立っている                                   0.552
    2: 落ち込んでいる                                  0.310
```

### 複数の質問をまとめて投げる

state の読み込みが 1 回で済むので効率的。

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "助けて！請求金額が3日前からずっと間違っています！" \
  --choice "どのチームが対応すべきですか？=請求,営業,技術" \
  --noul "緊急性を感じさせますか？" \
  --score "書き手はどれくらい苛立っていますか？=落ち着いている,苛立っている,落ち込んでいる"
```

**観察ポイント**: state や選択肢を変えると confidence がどう動くか。

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "AIが書いたブログが氾濫する世の中になってしまいました" \
  --choice "どうするべき？=自分もAIブログで対抗する,温もりあふれる人手ブログを極める,気にしない" \
  --noul "AIが書いたブログを一切の推敲なしに世の中に出していいですか？" \
  --score "AIブログを読んだ読者はどんな気持ちになりますか?=感情の変化なし,苛立つ,がっかりする"
```

## Step 2: HTTP サーバーとして起動する

```bash
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8000
```

別のターミナルから:

```bash
curl -s localhost:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "助けて！請求金額が3日前からずっと間違っています！",
    "questions": {
      "緊急か": {"type": "noul", "instructions": "緊急性を感じさせますか？"}
    }
  }'
```

出力例(t4g.xlarge、CPU):

```json
{
  "model": "strands-decider-2B-hobson-v19",
  "answers": {
    "緊急か": {
      "type": "noul",
      "noul": 0.7894
    }
  },
  "usage": {
    "input_tokens": 89,
    "output_tokens": 1
  },
  "latency_ms": 3257.89
}
```

**観察ポイント**: `latency_ms` で手元環境のレイテンシを測る。比較対象は公式値(RTX 3090 で中央値 115ms、M3 Pro で 153ms)。

## Step 3: Strands エージェントに組み込む

Amazon Bedrock の LLM を使うため、Bedrock にアクセスできる AWS 認証情報が必要。

> **`NoRegionError: You must specify a region.` が出る場合**: `aws login` で作った認証情報(`login_session`)は、期限が切れると botocore が自動で更新する。この更新用クライアントは `AWS_REGION` を見ず、`AWS_DEFAULT_REGION` かプロファイルの `region` を使う。`export AWS_DEFAULT_REGION=ap-northeast-1`(または `aws configure set region ap-northeast-1`)を設定しておく。

```bash
git clone https://github.com/strands-labs/strands-decider && cd strands-decider
uv pip install -e . strands-agents
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
```

別のターミナルから:

```bash
python examples/strands/tool_call_intervention.py
```

サンプルはポート 8099 のサーバーを使う。変えるときは `STRANDS_DECIDER_URL` で指定する。

### シナリオ

このサンプルだけは公式のコードなので、Decider への質問文が英語。Step 4 以降は日本語に直してある。

1. エージェントには `get_weather` ツールと、わざと「せっかち」にした system prompt が与えられている。
2. ユーザーが場所を言わずに「What's the weather?」と聞くと、エージェントは都市を推測して `get_weather` を呼ぼうとする。
3. ツールが実行される前に、`InterventionHandler.before_tool_call` の中で Decider が次の 2 つの Yes/No に答える。
   - 引数の値は、ユーザーが実際に言ったことに基づいているか
   - ユーザーに確認する前にこのツールを呼ぶのは早すぎないか
4. その結果をもとに `Guide` を返すと、エージェントは天気を答える代わりに都市を聞き返す。

```python
QUESTIONS = {
    "args_grounded": Decider.noul(
        "Are the tool's argument values grounded in facts the user actually provided?",
        {
            "true": "every argument value traces back to something the user said",
            "false": "an argument value was guessed or invented, not stated by the user",
        },
    ),
    "premature": Decider.noul(
        "Is it premature to call this tool now, before clarifying with the user?",
        {
            "true": "the assistant should ask a clarifying question before calling the tool",
            "false": "there is nothing left to clarify; calling now is appropriate",
        },
    ),
}
```

### Intervention の返り値

| アクション | 意味 |
| --- | --- |
| `Proceed` | ツールをそのまま実行する |
| `Deny` | ツール呼び出しを拒否する |
| `Confirm` | 止めて人間に確認する |
| `Guide` | フィードバックを付けてモデルにターンを返し、軌道修正させる |

**観察ポイント**: ユーザーの発言に都市名を含めたときに `Proceed` へ変わるか。しきい値を変えると挙動がどう変わるか。

## Step 4: LLM で回答するか、人間にエスカレーションするかを振り分ける

サポート問い合わせを、エージェントが LLM を呼ぶ**前**に Decider で振り分ける。人間に回す問い合わせでは LLM を一度も呼ばない。サンプルはハンズオンのルートにある `step4_escalation_gate.py`。

自前の制御フローは書かず、Strands の純正機能だけで分岐させる。

| 使う機能 | 役割 |
| --- | --- |
| `InterventionHandler.before_invocation` | エージェントがリクエストを処理し始める前(LLM 呼び出し前)に走るフック。ここで Decider に問い合わせる |
| `Proceed` | LLM 側の分岐。そのまま Bedrock の LLM が回答する |
| `Deny` | 人間側の分岐。invocation をキャンセルし、LLM は呼ばれない。`reason` がアシスタントの返答として会話に残る |
| `agent.state` | 判定結果(route、確率、confidence)を呼び出し側に渡す |

Decider への質問は `choice` 1 つ。「担当者」の確率が 0.7 以上なら人間に回す。

```python
QUESTIONS = {
    "route": Decider.choice(
        "このサポートへの問い合わせは、誰が対応すべきですか？",
        {
            "AIアシスタント": "製品ドキュメントをもとに、AIアシスタントだけで十分に答えられる一般的な質問。使い方、機能、プラン、...",
            "担当者": "人間のスタッフが必要な問い合わせ。返金、請求のトラブル、補償、アカウントのセキュリティ事故、...",
        },
    ),
}
```

Step 3 と同じくポート 8099 でサーバーを起動しておき、ハンズオンのルートで実行する。

```bash
python step4_escalation_gate.py                    # 組み込みのサンプル問い合わせ 5 件を順に流す
python step4_escalation_gate.py "返金してください"  # 任意の問い合わせを 1 件流す
```

実行例(抜粋。t4g.xlarge、CPU):

```text
USER: データをCSVで書き出すにはどうすればいいですか？
  [gate] route=llm p_human=0.106 confidence=0.787 decider=5457ms llm_input_tokens=86
  [llm] ASSISTANT: # CSVエクスポート方法 ...

USER: 注文番号1234で二重に請求されています。すぐに返金してください。
  [gate] route=human p_human=0.862 confidence=0.723 decider=5452ms llm_input_tokens=0
  [human] チケット起票: p_human=0.862 message='注文番号1234で二重に請求されています。すぐに返金してください。'
  ASSISTANT: 担当者におつなぎします。内容を確認のうえ、担当者からご連絡いたします。
```

- 人間側の分岐では `llm_input_tokens=0` になり、LLM が呼ばれていないことがわかる。
- `escalate_to_human()` はチケット起票を表示するだけ。実運用ではチケットシステムや Slack 通知に置き換える。
- Decider の判定は t4g.xlarge(CPU)で 1 回約 5 秒かかった。

**観察ポイント**:

- `パスワードを忘れました。どうすればいいですか？` は `confidence=0.046` と低く、`p_human=0.523` で LLM 側に振られる。しきい値 `HUMAN` を下げたり、「confidence が低ければ人間へ」という条件を足したりすると、振り分けがどう変わるか。
- 選択肢の説明文(`criteria`)を書き換えると確率がどう動くか。

## Step 5: どの LLM に答えさせるかを決める(モデルルーティング)

依頼の難しさを Decider で判定し、簡単なら Claude Haiku 4.5、難しいなら Claude Sonnet 4.6 に答えさせる。サンプルは `step5_model_router.py`。

| 使う機能 | 役割 |
| --- | --- |
| `ModelRouter` | 候補モデルの中から invocation ごとに 1 つを選んで使う。`Agent(model=router)` として渡す |
| `RoutingCandidate` | 候補モデルに名前(`軽量` / `高性能`)を付ける。先頭の候補が既定値 |
| `RoutingStrategy.select` | どの候補を使うかを決める非同期メソッド。ここで Decider に問い合わせる |

Strands 純正の `ClassifierStrategy` は、この判定自体を LLM に問い合わせる。Decider に置き換えると、判定がローカルの 1 回の forward pass で済む。`ModelRouter` は Strands 側で provisional(暫定)API とされている。

```python
QUESTIONS = {
    "model": Decider.choice(
        "この依頼は、どのモデルが対応すべきですか？",
        {
            "軽量": "小さくて速いモデルで十分な依頼。あいさつ、短い返答、フレーズの翻訳、...",
            "高性能": "推論力の高いモデルが必要な依頼。複数の手順にわたる分析、設計、デバッグ、計画、...",
        },
    ),
}

class DeciderStrategy:
    async def select(self, context, **kwargs):
        if context.attempts:  # 失敗後の再選択では切り替えない
            return None
        answer = (await asyncio.to_thread(self._decider.ask, latest_user_text(context), QUESTIONS))["model"]
        return next(c for c in context.candidates if c.name == answer["choice"])
```

```bash
python step5_model_router.py                                   # 組み込みのサンプル依頼 4 件を順に流す
python step5_model_router.py "この文を英訳して: 了解です"     # 任意の依頼を 1 件流す
```

実行例(抜粋。t4g.xlarge、CPU):

```text
USER: 「おはよう」をフランス語に訳して
  [router] 軽量 -> jp.anthropic.claude-haiku-4-5-20251001-v1:0 confidence=0.976 decider=4882ms
  [llm] total=6.0s input_tokens=72 output_tokens=67

USER: 2つのタスクが2つのロックを逆の順番で取得すると、非同期のPythonコードがデッドロックする原因を調べて、修正案を出して
  [router] 高性能 -> jp.anthropic.claude-sonnet-4-6 confidence=0.966 decider=5420ms
  [llm] total=20.0s input_tokens=111 output_tokens=1527
```

- 4 件とも意図どおりに振り分けられた(翻訳と言い換えは Haiku、デバッグとフェルミ推定は Sonnet)。
- 実際にどのモデルが使われたかは、ロガー `strands.models.routing` を INFO レベルにすると `candidate selected` のログで確認できる。
- `RoutingStrategy.select` は `async def` 必須。Decider クライアントは同期なので `asyncio.to_thread` で包む。

**観察ポイント**: 4 件とも confidence が 0.94 以上で、迷いなく振り分けられた。境界にありそうな依頼(短いが専門的な質問など)を投げると、どちらに振られるか。

## Step 6: どのエージェントに渡すかを決める(Graph の条件付きエッジ)

受付エージェントが問い合わせを 1 文の日本語チケットに要約し、Decider がそのチケットを請求・技術・営業の担当エージェントに振り分ける。ワークフローの形はコードで固定し、分岐の判断だけを Decider に任せる。サンプルは `step6_graph_routing.py`。

```text
受付 ──[Decider: 請求?]──> 請求担当
     ├─[Decider: 技術?]──> 技術担当
     └─[Decider: 営業?]──> 営業担当
```

| 使う機能 | 役割 |
| --- | --- |
| `GraphBuilder.add_node` | エージェントをノードとして登録する |
| `GraphBuilder.add_edge(condition=...)` | 条件関数が `True` を返したエッジだけを辿る。ここで Decider に問い合わせる |
| `GraphState.results` | 条件関数の中で、前のノード(受付)の出力を読む |

```python
@lru_cache(maxsize=128)
def route(ticket: str) -> str:
    return decider.ask(f"サポートチケット: {ticket}", QUESTIONS)["team"]["choice"]

def routed_to(team):
    def condition(state: GraphState) -> bool:
        return route(str(state.results["受付"].result).strip()) == team
    return condition

for team in TEAMS:
    builder.add_edge("受付", team, condition=routed_to(team))
```

```bash
python step6_graph_routing.py                                 # 組み込みのサンプル問い合わせ 3 件を順に流す
python step6_graph_routing.py "ログインすると500エラーが出ます"  # 任意の問い合わせを 1 件流す
```

実行例(抜粋。t4g.xlarge、CPU):

```text
USER: ログインすると500エラーが出ます
  [受付] ログイン時に500エラーが発生している。
  [route] 技術 confidence=0.993 decider=4379ms path=受付 -> 技術
  [技術] ログインに関する500エラーについてご報告いただきありがとうございます。...
```

- 3 件とも意図どおりの担当に届いた(confidence 0.67〜0.99)。
- 条件関数はエッジごとに呼ばれる(3 本なら 3 回)。同じ要約文の判定は `lru_cache` でキャッシュし、Decider の呼び出しを 1 問い合わせ 1 回に抑えている。
- 受付エージェントの要約も日本語にしている。Decider の学習データは英語中心だが、日本語でも判定できた。要約がそのまま社内チケットの件名にもなる。

**観察ポイント**: `TEAMS` に担当(例: `account`: ログイン・パスワード・アカウント設定)を足すと、ノードとエッジが増えるだけで振り分けが変わる。どの問い合わせの担当が入れ替わるか。

## Step 7: ツールを実行してよいかを決める(HumanInTheLoop)

エージェントのツール呼び出しを Decider が見て、読むだけの操作は自動で実行し、変更や外部送信を伴う操作だけ人間に承認を求める。サンプルは `step7_tool_approval.py`。ツールはすべて偽物で、メモリ上の辞書を読み書きするだけ。

| 使う機能 | 役割 |
| --- | --- |
| `HumanInTheLoop` | ツール実行前に人間の承認を挟む Strands 純正の intervention |
| `classifier=` | 承認が要るかを判定する関数。`BeforeToolCallEvent` を受け取り `ClassifierResult` を返す。ここで Decider に問い合わせる |
| `ask="stdio"` | 承認をターミナルで y/n 入力で受け付ける |

`classifier=True` にすると純正の LLM リスク判定が使われる。Decider に置き換えると、判定がローカルの 1 回の forward pass で済む。`ClassifierResult` は `strands.vended_interventions.hitl` からは import できず、`strands.vended_interventions.hitl.classifier` から import する。

```python
def decider_classifier(event, **kwargs) -> ClassifierResult:
    tool_use = event.tool_use
    state = f"ツール名: {tool_use['name']}\n入力: {json.dumps(tool_use['input'], ensure_ascii=False)}"
    p = decider.ask(state, QUESTIONS)["needs_approval"]["noul"]
    return ClassifierResult(requires_human_in_the_loop=p >= 0.5, reason=f"Decider p={p:.2f}")

agent = Agent(
    tools=[list_files, read_file, send_email, delete_file],
    interventions=[HumanInTheLoop(classifier=decider_classifier, ask="stdio")],
)
```

```bash
python step7_tool_approval.py   # 承認を求められたら y/n で答える
```

実行例(t4g.xlarge、CPU)。メール送信は `y`、ファイル削除は `n` と答えた:

```text
USER: docs フォルダのファイルを確認して README を要約し、alice@example.com にメールで送って。そのあと tmp/old.log を削除して。

  [decider] list_files p=0.06 -> auto (5260ms)
  [decider] read_file p=0.07 -> auto (5245ms)
  [decider] send_email p=0.74 -> ask human (6348ms)
Approve "send_email" — Decider p=0.74?
  Input: {"to": "alice@example.com", ...} (y/n): y
  [decider] delete_file p=0.78 -> ask human (5180ms)
Approve "delete_file" — Decider p=0.78?
  Input: {"path": "tmp/old.log"} (y/n): n
  [tool] send_email to=alice@example.com subject='README 要約'

remaining files: ['docs/CHANGELOG.md', 'docs/README.md', 'tmp/old.log']
```

- 読むだけの `list_files` / `read_file` は承認なしで実行され、`send_email` と `delete_file` だけ承認を求められた。`n` と答えた削除は実行されず、ファイルは残った。
- 承認プロンプトの `Input:` は `HumanInTheLoop` が `json.dumps` で表示するため、日本語は `\uXXXX` にエスケープされて見える。

**観察ポイント**: 選択肢の説明文(`criteria`)の書き方で結果が変わる。最初に(英語で)試した次の質問では、全社員宛ての `send_email` を「承認不要」(p=0.31)と判定した。

```python
Decider.noul("Should a human approve this tool call before it runs?", {
    "true": "the call is destructive, irreversible, sends data outside, moves money, or touches sensitive files",
    "false": "the call only reads or lists non-sensitive information and is safe to run automatically",
})
```

サンプルの質問のように「何かを変更するか、外部に送るか」を問い、`true` 側にメール送信を明記すると、10 件のテストケースがすべて正しく判定された。

## Step 8(任意): 自分のユースケースで試す

- ツール選択、ガードレール、ポリシー分類などの質問を自分で作る。
- 判定をエージェントのループに入れてもレイテンシが許容範囲に収まるかを測る。
- 話題に合ったポリシーだけを system prompt に差し込む(`ContextInjector` の `render_content` で Decider の `choice` を使う)。

## スコープ外

- **再学習**: Linux/WSL2 と NVIDIA GPU が必要。目安は RTX 3090(24GB)1 枚で約 11 時間。手順はリポジトリの `training/README.md` にある。
