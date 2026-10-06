# strands-decider-handson

[Strands Decider 2B](https://github.com/strands-labs/strands-decider) を CLI → HTTP サーバー → Strands エージェント組み込みの順に試すハンズオン。Lambda MicroVM(aarch64、CPU のみ)で全ステップを実際に動かして確認した。

## ファイル

| ファイル | 内容 |
| --- | --- |
| [HANDSON.md](HANDSON.md) | ハンズオン手順。前提環境、Step 1〜8 |
| [step4_escalation_gate.py](step4_escalation_gate.py) | Step 4。LLM で回答するか人間にエスカレーションするかを振り分ける(`InterventionHandler.before_invocation`) |
| [step5_model_router.py](step5_model_router.py) | Step 5。依頼の難しさで Claude Haiku 4.5 と Sonnet 4.6 を使い分ける(`ModelRouter`) |
| [step6_graph_routing.py](step6_graph_routing.py) | Step 6。問い合わせを請求・技術・営業の担当エージェントに振り分ける(`Graph` の条件付きエッジ) |
| [step7_tool_approval.py](step7_tool_approval.py) | Step 7。変更や外部送信を伴うツール呼び出しだけ人間に承認を求める(`HumanInTheLoop`) |

## クイックスタート

```bash
git clone https://github.com/har1101/strands-decider-handson && cd strands-decider-handson
uv venv -p 3.14 && source .venv/bin/activate
uv pip install strands-decider

# Lambda MicroVM などの aarch64 環境では必須(理由は HANDSON.md「推論が止まる場合」)
export NVPL_BLAS_DEBUG_CPU_TYPE=1

strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "Help! My payouts have been failing for 3 days!" \
  --choice "Which team should handle this?=billing,sales,retail"
```

Step 3〜7 は、公式リポジトリをこのリポジトリの直下に clone して使う。`step*.py` は `strands-decider/examples/strands/_client.py` の `Decider` クライアントを読み込むので、このディレクトリ構成が前提になる。

```text
strands-decider-handson/
├── HANDSON.md
├── README.md
├── step4_escalation_gate.py
├── step5_model_router.py
├── step6_graph_routing.py
├── step7_tool_approval.py
└── strands-decider/        # git clone https://github.com/strands-labs/strands-decider
```

## 動作確認した環境

| 項目 | 値 |
| --- | --- |
| 実行環境 | Lambda MicroVM(ベースライン 4GB / 2vCPU、ピーク 16GB / 8vCPU、ディスク 16GB) |
| CPU | Neoverse V1(Graviton3)、SVE 無効 |
| Python | 3.12 |
| torch / transformers / strands-agents | 2.14.1 / 5.18.0 / 1.57.2 |
| LLM(Step 3〜7) | Amazon Bedrock `jp.anthropic.claude-haiku-4-5-20251001-v1:0`、Step 5 のみ `jp.anthropic.claude-sonnet-4-6` も使う(`AWS_REGION=ap-northeast-1`) |

## わかったこと

### Lambda MicroVM で推論が止まる

- `Loading weights: 100%` のあと推論が終わらない。原因はメモリ不足ではなく、PyTorch 同梱の NVPL BLAS が SVE 無効の Neoverse V1 で無限ループすること。
- `export NVPL_BLAS_DEBUG_CPU_TYPE=1` で回避できる。詳細と確認コマンドは HANDSON.md の「推論が止まる場合(NVPL BLAS の回避策)」にある。
- 実行中のメモリはピークで約 12.4GB。ディスクは 16GB 中 15GB が埋まった(残り約 600MB)。MicroVM ではベースライン 4GB / 2vCPU(ディスク 16GB)以上が必要。
- 回避策を入れると、Step 1 の `ask` 1 回(モデル読み込み込み)が約 9 秒、サーバー経由の判定 1 回が約 1.4〜2.9 秒だった。

### 結果は決定的、ただし confidence が低い答えは文言で揺れる

- 同じ入力なら結果は小数点以下まで一致する。生成ループを持たないため、実行ごとのランダム性はない。
- 正解のない意見を聞く質問では confidence が 0.17〜0.59 に留まった。日本語と英語で同じ内容を聞くと答えが入れ替わることもあった。
- 実運用では confidence にしきい値を設け、下回ったら LLM や人間に回す。

### Step 4〜7: Decider の判定を Strands の純正機能に差し込む

どのステップも、Decider に判定させる関数を 1 つ書いて Strands の純正機能に渡すだけで済む。

| Step | Decider が決めること | 差し込む先 | 質問の型 | 結果 |
| --- | --- | --- | --- | --- |
| 4 | LLM で答えるか、人間に回すか | `InterventionHandler.before_invocation` | choice | 5/5 |
| 5 | Haiku と Sonnet のどちらに答えさせるか | `ModelRouter` の `RoutingStrategy` | choice | 4/4 |
| 6 | 請求・技術・営業のどの担当に渡すか | `GraphBuilder.add_edge(condition=...)` | choice | 4/4 |
| 7 | ツール呼び出しに人間の承認が要るか | `HumanInTheLoop(classifier=...)` | noul | 4/4 |

- 結果は、各サンプルを Bedrock 込みで実行したときに意図どおりに判定された件数。判定 1 回は CPU で約 1.4〜2.3 秒だった。
- Step 4 では、人間に回した問い合わせで Bedrock の入力トークンが 0 になり、LLM を呼んでいないことを確認した。ただし `パスワードを忘れました…` は confidence 0.074 と低く、きわどい判定だった。
- Step 7 では、選択肢の説明文の書き方で結果が変わった。最初の書き方では全社員宛ての `send_email` を承認不要と判定したが、「変更するか、外部に送るか」を問う形に直すと 10 件のテストケースがすべて正しく判定された。
- 質問文と選択肢の説明は英語で書いた。入力が日本語でも、今回の判定はすべて英語の質問文で通している。

### `aws login` の認証情報で `NoRegionError` が出る

- `aws login` の認証情報は期限が切れると botocore が自動更新するが、更新用クライアントは `AWS_REGION` を見ない。
- `export AWS_DEFAULT_REGION=ap-northeast-1` を設定しておくと避けられる。

## 参考

- ブログ: https://strandsagents.com/blog/introducing-strands-decider/
- GitHub: https://github.com/strands-labs/strands-decider
- 重み: https://huggingface.co/StrandsAgents
