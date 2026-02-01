# 謎の国家 — RAG 精度検証・比較ツール

架空の国家の法体系を題材にした **RAG (Retrieval-Augmented Generation)** システム。
AI による司法判断の精度を検証・比較するためのツールキットです。

## 概要

「謎の国家」は、独自の憲法・刑法・民法・行政法・文化規制・倫理指針を持つ架空の国家です。
本プロジェクトは、この法体系と 1,206 件の判例データベースを知識ベースとして、LLM が法律質問に対して構造化された司法判断を生成し、その精度を評価するシステムを提供します。

### 主な機能

- **法律 Q&A チャット** — 法律に関する質問に対し、条文引用・判例参照付きで司法判断を生成
- **エージェントモード** — ReAct エージェントがツールを自律的に選択・実行して法的質問に回答
- **書庫アクセスツール** — 法令検索・判例検索・書庫統計の 3 つの LangChain Tool を提供
- **複数モデル精度比較** — 異なる LLM 設定間で回答精度を定量的に比較
- **テストケース管理** — 検証用のテストケースを作成・管理（CRUD）
- **ベクトル検索** — ChromaDB を用いた法的文書・判例の類似度検索

## プロジェクト構成

```
nazono-kokka/
├── app.py                    # Streamlit エントリーポイント
├── requirements.txt          # Python 依存パッケージ
├── rag_system/               # RAG コアシステム
│   ├── config.py             # 設定（パス, LLM, Embedding, チャンク等）
│   ├── main.py               # CLI インターフェース
│   ├── ingest.py             # ドキュメント取り込みパイプライン
│   ├── retriever.py          # ベクトル検索・取得ロジック
│   ├── judge.py              # 司法推論チェーン（LangChain）
│   ├── tools.py              # LangChain Tool 定義（法令検索・判例検索・書庫統計）
│   ├── agent.py              # ReAct エージェント（ツール統合）
│   ├── backend_adapter.py    # LLM バックエンド抽象化
│   ├── comparison_engine.py  # 複数モデル精度比較エンジン
│   ├── metrics.py            # 評価メトリクス算出
│   └── test_cases.py         # テストケース管理
├── ui/                       # Streamlit Web UI
│   ├── chat_page.py          # チャットページ
│   ├── settings_page.py      # 設定ページ
│   ├── testcase_page.py      # テストケース管理ページ
│   ├── comparison_page.py    # 精度比較ダッシュボード
│   └── components.py         # 共通 UI コンポーネント
├── legal_framework/          # 法体系ドキュメント（Markdown）
│   ├── constitution.md       # 憲法（約 30 KB）
│   ├── criminal_code.md      # 刑法（約 75 KB）
│   ├── civil_code.md         # 民法（約 79 KB）
│   ├── administrative_code.md # 行政法（約 49 KB）
│   ├── cultural_regulations.md # 文化規制（約 25 KB）
│   └── ethical_guidelines.md # 倫理指針（約 25 KB）
├── precedents/               # 判例データベース（JSON, 全 1,206 件）
│   ├── metadata.json         # 判例メタデータインデックス
│   ├── criminal/             # 刑事判例（486 件）
│   ├── civil/                # 民事判例（510 件）
│   └── constitutional/       # 憲法判例（210 件）
├── test_cases/               # テストケース定義・結果
│   ├── default_cases.json    # デフォルトテストケース（15 件以上）
│   └── results/              # 比較結果出力先
├── tests/                    # ユニットテスト・統合テスト
└── scripts/                  # データ生成・検証スクリプト
    ├── generate_legal_framework.py
    ├── generate_precedents.py
    ├── validate_data.py
    └── verify_e2e.sh
```

## 技術スタック

| レイヤー | 技術 |
|----------|------|
| LLM バックエンド | [Ollama](https://ollama.ai)（ローカル実行） |
| ベクトル DB | [ChromaDB](https://www.trychroma.com/) |
| Embedding | HuggingFace `paraphrase-multilingual-mpnet-base-v2` |
| RAG フレームワーク | [LangChain](https://www.langchain.com/) |
| Web UI | [Streamlit](https://streamlit.io/) |
| 可視化 | [Plotly](https://plotly.com/) |
| テスト | pytest |

## セットアップ

### 前提条件

- Python 3.11 以上
- [Ollama](https://ollama.ai) がインストール済み
- jq（E2E 検証スクリプト用、任意）

### 1. リポジトリのクローン

```bash
git clone https://github.com/shigenoburyuto/nazono-kokka.git
cd nazono-kokka
```

### 2. 仮想環境の作成と依存パッケージのインストール

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Ollama のセットアップ

Ollama サーバーを起動し、使用するモデルをダウンロードします。

```bash
# Ollama サーバー起動
ollama serve

# 推奨モデルのダウンロード（別ターミナルで実行）
ollama pull schroneko/llama-3.1-swallow-8b-instruct-v0.1

# または汎用モデル
ollama pull llama3.1:8b
```

### 4. ドキュメントの取り込み（ChromaDB 初期化）

法体系ドキュメントと判例をベクトル DB に取り込みます。

```bash
source .venv/bin/activate
python rag_system/ingest.py
```

既存データをリセットして再取り込みする場合：

```bash
python rag_system/ingest.py --reset
```

## 起動方法

### Web UI（Streamlit）

```bash
source .venv/bin/activate
streamlit run app.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

ポートを変更する場合：

```bash
streamlit run app.py --server.port 8502
```

ヘッドレスモード（ブラウザを自動で開かない）：

```bash
streamlit run app.py --server.headless true
```

Web UI は 4 つのページで構成されています：

| ページ | 機能 |
|--------|------|
| **チャット** | 法律に関する質問を入力し、条文引用・判例参照付きの司法判断を取得。エージェントモード切替対応 |
| **設定** | 使用するモデル、temperature、コンテキスト長などのパラメータを設定 |
| **テストケース** | 検証用テストケースの作成・編集・削除 |
| **精度比較** | 複数の LLM 設定で同じテストケースを実行し、精度を比較するダッシュボード |

### CLI（コマンドライン）

```bash
source .venv/bin/activate

# 単発クエリ
python rag_system/main.py --query "窃盗罪の量刑基準を示せ"

# 対話モード（REPL）
python rag_system/main.py
```

## エージェントモード

チャットページでは **通常モード** と **エージェントモード** を切り替えられます。

### 通常モード（RetrievalQA チェーン）

従来の RAG パイプラインで、ベクトル検索 → LLM 推論の単一パスで回答を生成します。

### エージェントモード（ReAct エージェント）

LLM が自律的にツールを選択・実行して回答を構築します。複数ステップの推論が必要な複雑な法的質問に有効です。

エージェントが使用できるツール：

| ツール名 | 機能 | 主なパラメータ |
|----------|------|---------------|
| `legal_framework_search` | 法令データベース（憲法・刑法・民法等）を検索 | `query`, `k`（取得件数） |
| `precedent_search` | 判例データベースを検索 | `query`, `k`, `case_type`（刑事/民事/憲法）, `verdict`（有罪/無罪） |
| `archive_stats` | 書庫の統計情報を取得 | なし |

### プログラムからの利用

```python
from rag_system.agent import create_agent, run_agent

# エージェント作成
agent = create_agent()

# クエリ実行
result = run_agent(agent, "窃盗罪の量刑基準を示せ")

print(result["result"])        # 最終回答
print(result["tool_calls"])    # 使用されたツール一覧
```

ツール単体での利用：

```python
from rag_system.tools import legal_framework_search, precedent_search, archive_stats

# 法令検索
result = legal_framework_search.invoke({"query": "窃盗罪の構成要件"})

# 判例検索（フィルタ付き）
result = precedent_search.invoke({
    "query": "窃盗",
    "case_type": "criminal",
    "verdict": "有罪",
})

# 書庫統計
stats = archive_stats.invoke({})
```

## データ検証

### E2E 検証（シェルスクリプト）

Python 不要で、プロジェクト全体の整合性を検証します。

```bash
bash scripts/verify_e2e.sh
```

検証内容：
1. 法体系ドキュメントの存在・サイズ確認（6 ファイル、各 1KB 以上）
2. 判例数の確認（合計 1,000 件以上）
3. 全判例 JSON のパース検証
4. 判例スキーマ準拠（必須フィールドの存在確認）
5. metadata.json の整合性
6. Python モジュール構造の確認
7. 依存パッケージの確認

### Python によるデータ検証

```bash
source .venv/bin/activate
python scripts/validate_data.py
```

### テストの実行

```bash
source .venv/bin/activate
python -m pytest tests/ -x
```

## 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama サーバーの URL |
| `CHROMA_DB_PATH` | `<project_root>/chroma_db` | ChromaDB の保存先パス |
| `LLM_TEMPERATURE` | `0.1` | LLM の temperature |
| `LLM_NUM_CTX` | `4096` | LLM のコンテキストウィンドウサイズ |
| `CHUNK_SIZE` | `1000` | ドキュメントチャンクサイズ（文字数） |
| `CHUNK_OVERLAP` | `200` | チャンク間のオーバーラップ（文字数） |
| `RETRIEVAL_K` | `5` | 検索時の取得ドキュメント数 |

## 判例データベース

全 1,206 件の判例が 3 カテゴリに分類されています。

| カテゴリ | 件数 | Case ID 形式 | 主な事件種別 |
|----------|------|-------------|-------------|
| 刑事 | 486 | `CRIM-YYYY-NNNN` | 窃盗、殺人、詐欺、薬物犯罪、サイバー犯罪、強盗、傷害 等 |
| 民事 | 510 | `CIVIL-YYYY-NNNN` | 不法行為、売買契約、賃貸借、相続、家事事件、債務不履行 等 |
| 憲法 | 210 | `CONST-YYYY-NNNN` | 法の下の平等、表現の自由、プライバシー権、適正手続 等 |

## ライセンス

本プロジェクトに含まれる法体系・判例データはすべて架空のものであり、現実の法律とは関係ありません。
