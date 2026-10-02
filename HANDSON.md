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
uv venv -p 3.12
source .venv/bin/activate
uv pip install strands-decider
```

## Step 1: CLI で 3 種類の質問を試す

AWS は不要。初回実行時に Hugging Face からモデルがダウンロードされる。

### choice(選択)

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "Help! My payouts have been failing for 3 days!" \
  --choice "Which team should handle this?=billing,sales,retail"
```

公式 README の出力例:

```text
choice_0 -> billing (confidence 0.768)
  billing                  0.845
  retail                   0.091
  sales                    0.064
```

### noul(Yes/No)

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "Help! My payouts have been failing for 3 days!" \
  --noul "Does this convey urgency?"
```

```text
noul_0 noul = 0.828
```

### score(採点)

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "Help! My payouts have been failing for 3 days!" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```

```text
score_0 score = 1.10 (confidence 0.518)
  0: calm                                     0.163
  1: frustrated                               0.573
  2: depressed                                0.265
```

### 複数の質問をまとめて投げる

state の読み込みが 1 回で済むので効率的。

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "Help! My payouts have been failing for 3 days!" \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
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
    "state": "Help! My payouts have been failing for 3 days!",
    "questions": {
      "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
    }
  }'
```

公式 README の出力例:

```json
{
  "model": "strands-decider-2B-hobson-v19",
  "answers": {
    "is_urgent": {
      "type": "noul",
      "noul": 0.8277
    }
  },
  "usage": {
    "input_tokens": 86,
    "output_tokens": 1
  },
  "latency_ms": 140.03
}
```

**観察ポイント**: `latency_ms` で手元環境のレイテンシを測る。比較対象は公式値(RTX 3090 で中央値 115ms、M3 Pro で 153ms)。

## Step 3: Strands エージェントに組み込む

Amazon Bedrock の LLM を使うため、Bedrock にアクセスできる AWS 認証情報が必要。

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

## Step 4(任意): 自分のユースケースで試す

### 例: LLM で回答するか、人間にエスカレーションするかを振り分ける

サポート問い合わせを、エージェントが LLM を呼ぶ**前**に Decider で振り分ける。人間に回す問い合わせでは LLM を一度も呼ばない。サンプルはハンズオンのルートにある `step4_escalation_gate.py`。

自前の制御フローは書かず、Strands の純正機能だけで分岐させる。

| 使う機能 | 役割 |
| --- | --- |
| `InterventionHandler.before_invocation` | エージェントがリクエストを処理し始める前(LLM 呼び出し前)に走るフック。ここで Decider に問い合わせる |
| `Proceed` | LLM 側の分岐。そのまま Bedrock の LLM が回答する |
| `Deny` | 人間側の分岐。invocation をキャンセルし、LLM は呼ばれない。`reason` がアシスタントの返答として会話に残る |
| `agent.state` | 判定結果(route、確率、confidence)を呼び出し側に渡す |

Decider への質問は `choice` 1 つ。`human_agent` の確率が 0.7 以上なら人間に回す。

```python
QUESTIONS = {
    "route": Decider.choice(
        "Who should handle this customer support message?",
        {
            "ai_assistant": "a general question an AI assistant can fully answer from product documentation: ...",
            "human_agent": "needs a human staff member: refunds, billing disputes, compensation, account security incidents, legal threats, ...",
        },
    ),
}
```

Step 3 と同じくポート 8099 でサーバーを起動しておき、ハンズオンのルートで実行する。

```bash
python step4_escalation_gate.py                    # 組み込みのサンプル問い合わせ 5 件を順に流す
python step4_escalation_gate.py "返金してください"  # 任意の問い合わせを 1 件流す
```

Lambda MicroVM(CPU)での実行例(抜粋):

```text
USER: How do I export my data as CSV?
  [gate] route=llm p_human=0.092 confidence=0.817 decider=1569ms llm_input_tokens=51
  [llm] ASSISTANT: I'd be happy to help you export your data as CSV! ...

USER: I was charged twice for order #1234. Refund me now.
  [gate] route=human p_human=0.964 confidence=0.928 decider=1515ms llm_input_tokens=0
  [human] チケット起票: p_human=0.964 message='I was charged twice for order #1234. Refund me now.'
  ASSISTANT: 担当者におつなぎします。内容を確認のうえ、担当者からご連絡いたします。
```

- 人間側の分岐では `llm_input_tokens=0` になり、LLM が呼ばれていないことがわかる。
- `escalate_to_human()` はチケット起票を表示するだけ。実運用ではチケットシステムや Slack 通知に置き換える。
- Decider の判定は CPU で 1 回約 1.5 秒かかった。

**観察ポイント**:

- `パスワードを忘れました。どうすればいいですか？` は `confidence=0.074` と低く、`p_human=0.463` で LLM 側に振られる。しきい値 `HUMAN` を下げたり、「confidence が低ければ人間へ」という条件を足したりすると、振り分けがどう変わるか。
- 選択肢の説明文(`criteria`)を書き換えると確率がどう動くか。

### 発展: ツール呼び出し単位で人間の承認を挟む

Strands には、ツール呼び出しの前に人間の承認を挟む純正の `HumanInTheLoop`(`strands.vended_interventions.hitl`)がある。`classifier` に任意の関数(`BeforeToolCallEvent` を受け取り `ClassifierResult` を返す)を渡せるので、組み込みの LLM リスク判定の代わりに Decider を使える。承認待ちは interrupt/resume(デフォルト)か `ask="stdio"` で受け付ける。

### その他のアイデア

- ツール選択、モデルルーティング、ガードレールなどの質問を自分で作る。
- 判定をエージェントのループに入れてもレイテンシが許容範囲に収まるかを測る。

## スコープ外

- **再学習**: Linux/WSL2 と NVIDIA GPU が必要。目安は RTX 3090(24GB)1 枚で約 11 時間。手順はリポジトリの `training/README.md` にある。
