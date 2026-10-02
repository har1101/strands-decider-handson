# strands-decider-handson

[Strands Decider 2B](https://github.com/strands-labs/strands-decider) を CLI → HTTP サーバー → Strands エージェント組み込みの順に試すハンズオン。Lambda MicroVM(aarch64、CPU のみ)で全ステップを実際に動かして確認した。

## ファイル

| ファイル | 内容 |
| --- | --- |
| [HANDSON.md](HANDSON.md) | ハンズオン手順。前提環境、Step 1〜4 |
| [step4_escalation_gate.py](step4_escalation_gate.py) | Step 4 のサンプル。LLM で回答するか人間にエスカレーションするかを Decider で振り分ける |

## クイックスタート

```bash
git clone https://github.com/har1101/strands-decider-handson && cd strands-decider-handson
uv venv -p 3.12 && source .venv/bin/activate
uv pip install strands-decider

# Lambda MicroVM などの aarch64 環境では必須(理由は HANDSON.md「推論が止まる場合」)
export NVPL_BLAS_DEBUG_CPU_TYPE=1

strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "Help! My payouts have been failing for 3 days!" \
  --choice "Which team should handle this?=billing,sales,retail"
```

Step 3 と Step 4 は、公式リポジトリをこのリポジトリの直下に clone して使う。`step4_escalation_gate.py` は `strands-decider/examples/strands/_client.py` の `Decider` クライアントを読み込むので、このディレクトリ構成が前提になる。

```text
strands-decider-handson/
├── HANDSON.md
├── README.md
├── step4_escalation_gate.py
└── strands-decider/        # git clone https://github.com/strands-labs/strands-decider
```

## 動作確認した環境

| 項目 | 値 |
| --- | --- |
| 実行環境 | Lambda MicroVM(ベースライン 4GB / 2vCPU、ピーク 16GB / 8vCPU、ディスク 16GB) |
| CPU | Neoverse V1(Graviton3)、SVE 無効 |
| Python | 3.12 |
| torch / transformers / strands-agents | 2.14.1 / 5.18.0 / 1.57.2 |
| LLM(Step 3・4) | Amazon Bedrock `jp.anthropic.claude-haiku-4-5-20251001-v1:0`(`AWS_REGION=ap-northeast-1`) |

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

### Step 4: LLM か人間かを Strands の純正機能で振り分ける

- `InterventionHandler.before_invocation` で LLM 呼び出し前に Decider に判定させ、`Proceed`(LLM が回答)か `Deny`(LLM を呼ばずに人間へ)を返す。
- 人間に回した問い合わせでは Bedrock の入力トークンが 0 で、LLM を呼んでいないことを確認した。
- サンプル 5 件(英語・日本語の一般質問、二重請求、不正ログイン)は意図どおりに振り分けられた。ただし `パスワードを忘れました…` は confidence 0.074 と低く、きわどい判定だった。
- ツール呼び出し単位で人間の承認を挟みたい場合は、Strands 純正の `HumanInTheLoop`(`strands.vended_interventions.hitl`)の `classifier` に Decider を渡す構成が使える(未検証)。

## 参考

- ブログ: https://strandsagents.com/blog/introducing-strands-decider/
- GitHub: https://github.com/strands-labs/strands-decider
- 重み: https://huggingface.co/StrandsAgents
