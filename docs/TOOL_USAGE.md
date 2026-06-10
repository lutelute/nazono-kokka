# 類似度ツール 使い方

`highlight_tool.py` ― 謎の国家 RAG の埋め込みを使って「似ている箇所」と
「書庫の意味軸」を見るスタンドアロン Streamlit ツール。

## 起動

```bash
streamlit run highlight_tool.py
```

ブラウザが自動で開く（`http://localhost:8501`）。このツールは新規ファイルなので、
既存の `app.py` / `ui/visualization_page.py` には一切手を入れていない。

## 機能

### 🎨 似ている箇所ハイライト
2つのテキスト（プリセット6種＋自由入力）の「どこが似ているか」を3方式で色付け。
コサイン類似度という **1つの数字** が、文章のどこから来たのかを分解して見せる。

| 方式 | 内容 |
|---|---|
| 📑 文アライメント | 対応する文を同色で結ぶ（日本語で最も読みやすい） |
| 🔥 トークンヒートマップ | 各語の寄与を暖色(似)/寒色(離)で。中心化コントラストで助詞を抑制 |
| 🔗 トークンペアリング | 似た語を同色＋番号で結ぶ（BERTScore 的 max-sim） |

### 🧭 書庫の意味軸
ChromaDB の全文書埋め込み（768次元）から意味のある軸を抽出。

- **PCA 説明分散比**：実質いくつの軸に情報が乗っているか（上位20軸で約52%）
- **概念軸**：任意の2グループ（例：法令系 ⇄ 判例系）の重心差で対比の向きを作り、
  全文書を射影してグループ別の分布（平均・標準偏差）を見る

## 必要なもの

- 依存：`streamlit` / `sentence-transformers` / `scikit-learn` / `chromadb`
  （`requirements.txt` に含まれる）
- 🧭 軸タブは **取り込み済みの ChromaDB**（`chroma_db/`）が必要。
  無い場合は先に `python -m rag_system.ingest` を実行する。

## 仕組みの解説（あわせて参照）

- `docs/infographic.html` … 概念の図解（矢印・厳密分解・PCA・概念軸）
- `docs/result.html` … 桃太郎×浦島太郎の実データ結果
- `docs/infographic_prompt.md` … 図解インフォグラフを生成する再利用プロンプト

## 内部構成（再利用しているモジュール）

- `rag_system/text_alignment.py` … 文/トークン寄与/ペアリングの計算（mean-pooling 厳密分解）
- `ui/highlight_view.py` … 3方式の描画（`render_similarity_highlight`）
- `rag_system/semantic_axes.py` … PCA分散比・概念軸・投影分散
