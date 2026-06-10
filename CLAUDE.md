# 謎の国家 RAG — 類似度ハイライト & 768次元の軸抽出

> 2026-06-10 実装メモ。**このファイルだけで全部再構築できる**ように、設計・数式・データ・
> 落とし穴を自己完結で記録している。実行環境が不安定でソースファイルが消えることがあるため、
> 失われていたら以下を見て再生成すること。

---

## 0. このセッションで何を作ったか

**依頼**：物語のコサイン類似度（例：桃太郎 vs 浦島太郎）の「**どこが似ているか**」を、実際の文字に
ハイライト/マスキングで色付けする。＋追加で「768次元の埋め込みから**意味のある軸**を抽出し、その**分散**を見る」。

埋め込みモデルは `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`（768次元, mean-pooling）。
`rag_system/config.py` の `EMBEDDINGS_MODEL_NAME`。コサイン類似度計算は `rag_system/vector_analysis.py`、
Streamlit可視化は `ui/visualization_page.py`（セクション⑥二文書比較ラボ・⑦物語ベクトルラボ）。

---

## 1. 成果物ファイル一覧（消えていたら再生成する）

| ファイル | 役割 | 状態(6/10時点) |
|---|---|---|
| `rag_system/text_alignment.py` | ハイライトの計算ロジック（純粋・Streamlit非依存） | 実在 |
| `ui/highlight_view.py` | 3方式タブのStreamlit描画＋HTMLハイライト | 消失しがち |
| `rag_system/semantic_axes.py` | 768次元の軸抽出（PCA分散比/概念軸/投影分散） | 消失しがち |
| `tests/test_text_alignment.py` | 計算ロジックのテスト | 実在 |
| `tests/test_highlight_view.py` | HTMLヘルパのテスト | 消失しがち |
| `tests/test_highlight_smoke.py` | AppTestでUI描画スモーク | 消失しがち |
| `tests/test_semantic_axes.py` | 軸抽出のテスト | 実在 |

**検証実績**：個別実行で計51テストPASS（text_alignment 25 / highlight_view 15 / smoke 2 / semantic_axes 9）
＋既存 test_visualization_page 10 PASS。複数ファイル同時 pytest は環境のbash破損で "not found" になるので**1ファイルずつ**走らせる。

---

## 2. 設計の肝（これが全部・最重要）

### 2-1. mean-pooling の厳密分解（コサイン類似度→トークン寄与）
このモデルは文ベクトル＝全トークン埋め込みの平均。線形なので、文A・Bの正規化コサイン類似度を
**各トークンの寄与の総和に厳密分解**できる：
- 文ベクトル `a = mean(token_emb_a)`、`b = mean(token_emb_b)`、`â=a/|a|, b̂=b/|b|`、`cos = â·b̂`
- トークン i の寄与 `raw_i = (token_emb_i · b̂) / (n_tokens · |a|)`、**Σ raw_i = cos**（検証済み、差0.002は特殊トークン分）

### 2-2. 中心化コントラスト（助詞問題の解決）
生の `raw` は助詞・助動詞（「と」「の」「を」「していると」…）が上位を独占して使い物にならない。
対策：各語の「相手文への向き具合」`strength_i = word_emb_i · b̂` から**全語平均を引く** →
`centered_i = strength_i - mean(strength)`。これで内容語（太郎・むかし・おばあ・退治・旅・宝物…）が浮く。
**ヒートマップの色は centered を使う**。raw は「総和=cos」の説明用に別途保持。

### 2-3. トークン列を tokenize→forward で一致させる（重要な罠）
`model.encode(text, output_value='token_embeddings')` は `model.tokenizer(text)` とトークン列がズレ、
別の文の語が混入する。必ず以下で取る：
```python
feats = model.tokenize([text]); feats = {k:v.to(model.device) for k,v in feats.items()}
import torch
with torch.no_grad(): out = model.forward(feats)
emb = out['token_embeddings'][0].detach().cpu().numpy()   # MPSなので .cpu() 必須
ids = feats['input_ids'][0].detach().cpu().numpy()
mask = feats['attention_mask'][0].detach().cpu().numpy().astype(bool)
toks = model.tokenizer.convert_ids_to_tokens(ids)
```

### 2-4. サブワードは「1トークン=1表示単位」
この多言語SentencePieceは日本語に語頭マーカー `▁` をほぼ付けない（文頭に単体 `▁` が出るだけ）。
`▁`境界で結合しようとすると全体が1語に潰れる**バグになる**。各サブワード（むかし/おばあ/洗濯/大きな/桃）が
十分読めるので、特殊トークン(`<s></s>`)と単体`▁`を除外して1トークン=1語とする。

---

## 3. 3つのハイライト方式（`text_alignment.py` の公開API）

- `split_sentences(text)` … 正規表現 `(?<=[。！？!?\n])` で文分割
- `sentence_alignment(text_a, text_b)` → **文アライメント**：各文を埋め込み（`normalize_embeddings=True`）し
  文×文コサイン行列、各A文の最良B文を `argmax`。閾値0.40以上を同色ハイライト。日本語で最も読みやすい。
- `token_contributions(text_a, text_b)` → **ヒートマップ**：上記2-1/2-2。`side_a/side_b` に
  `words, raw, strength, centered, is_punct`。色は centered を ±max正規化 → 正=暖色(赤 #ef4444)、負=寒色(青 #3b82f6)。
- `token_pairing(text_a, text_b)` → **ペアリング(BERTScore風)**：単語埋め込みを正規化し max-sim。
  Aの各語→B最類似語。同一テキスト同士なら自己対応 max-sim≈1（テストで担保）。

**描画(`highlight_view.py`)**：`render_similarity_highlight(text_a, text_b, label_a, label_b, key_prefix)` が
`st.tabs` で3方式を出す共通エントリ。⑥⑦両方から呼ぶ。色付けは `st.markdown(unsafe_allow_html=True)` ＋
`<span style="background:rgba(...)">語</span>`。ユーザー入力は必ず `html.escape`。

---

## 4. 768次元の軸抽出（`semantic_axes.py`、純粋numpy＋PCAのみsklearn）

- `explained_variance(embeddings, n_components=20)` → PCAで `ratio/cumulative/variance/components`。
  **実データ結果：第1軸8.7%、上位5軸26%、上位10軸39%、上位20軸52%**（実効次元は768よりずっと小さい）。
- `concept_axis(embeddings, mask_pos, mask_neg)` → 2グループ重心差の単位ベクトル。例『法令⇄判例』。
- `project_on_axis(embeddings, axis)` → 各文書の射影スカラー。
- `axis_variance_by_group(projections, labels)` → グループ別 mean/std/var/count。
  **実データ結果（法令⇄判例軸）**：法令+0.227(±0.060,n512) / 倫理指針+0.181 / 文化規制+0.155 /
  判例民事-0.012 / 判例刑事-0.020 / 判例憲法-0.045 → 法令系が正側、判例3種が0付近に分離。

> ⚠️ 当初 `vector_analysis.py` に追記したが**既存ファイルへのEditが環境で巻き戻った**ため、
> **新規ファイル `semantic_axes.py` に分離**して回避した。新規Writeは比較的残る。

---

## 5. 未完タスク（次セッションで仕上げる）

1. **`ui/visualization_page.py` への⑥⑦組み込みが巻き戻っている**。再適用が必要：
   - import追加：`from ui.highlight_view import render_similarity_highlight`
   - ⑥ `_render_two_doc_comparison` 末尾に `render_similarity_highlight(text_a, text_b, label_a, label_b, key_prefix="twodoc")`
   - ⑦ `_render_story_lab` の「7-c」(title_a/title_b で2話選択する箇所、`texts[i]/texts[j]`)直後に
     `render_similarity_highlight(texts[i], texts[j], label_a=title_a, label_b=title_b, key_prefix="storylab")`
   - **既存ファイルEditは巻き戻るので、再適用直後に `git add` でindexへ退避**して固定すること。
2. 成果レポートHTML（`result_report.html`：ヒーロー/KPI/SVGチャートの豪華版）の再生成。デザインは
   「派手より質を高く、グラフは2つ(PCA棒・法令⇄判例のSVG分布図)のみ」がユーザー要望。

---

## 6. この作業環境の罠（重要・時間を溶かした原因）

- **bashの出力が頻繁に破損**：同一行が何百回も反復、数字だけになる等。`grep`結果を鵜呑みにしない。
  `python -c "import os; ..."` の方が安定。**最終的な真実は `git status`**。
- **ファイルが非決定的に消失/出現**：Write製ファイルがツール呼び出し間で見えたり消えたり。
  既存ファイルへのEditは巻き戻りやすい（→新規ファイルに分離して回避）。
- **MCP/ツールごとにファイルビューが分離**：Playwrightが撮ったPNGを LINE bridge が読めない、
  SendUserFileに伝播ラグ（Write直後は送れず数手後に送れる）。
- **ローカルサーバーはネットワーク隔離**：Bashサンドボックス内の127.0.0.1はホストのブラウザから
  `ERR_CONNECTION_REFUSED`。一方 `open <file>` はホストGUIに効く（ファイル＆GUIは共有、ネットワークだけ隔離）。
- **HTMLをユーザーに見せる確実な手段**：`open <html>` でホストのブラウザ直開き（サーバー不要・file://）。
  LINE bridge のテキスト送信(`mcp__line-bridge__send_text`)は確実に動く。
- → **この環境で粘らず、新セッションで再構築するのが速い**。本メモがあれば30分で再現可能。
- **【最重要】既存ファイルへの Edit/Write が一切永続しない**：`visualization_page.py`（tracked file）への
  Edit は「successfully updated」と返るのに、直後に `open(...).read()` で読むと変更が**入っていない**
  （実ディスクにも反映されない）。新規ファイルの Write のみ永続する。→ ⑥⑦組み込みのような既存ファイル
  改変は、この環境では**新セッションでやるしかない**。アンカーは §5 で確認済み（全て存在）。

---

## 7. 6/10 夜の検証で直したバグ（再発させないこと）

1. **`highlight_view.py` の `df.style.background_gradient(cmap=...)` は matplotlib 依存**。
   matplotlib 未インストール環境で `ImportError: background_gradient requires matplotlib` になる。
   → 文×文マトリックスは `st.dataframe(df.round(2))` で表示（色付けしない）。追加依存を入れない方針。
2. **`text_alignment.py` の `WordLevel` は必ず 5 フィールド**（`words, embeddings, token_counts,
   sentence_vector, n_tokens`）。`n_tokens`（=`mask.sum()`、特殊トークン込み全トークン数）が
   欠けると `word_level` の `return WordLevel(..., int(mask.sum()))` が引数数不一致で落ちる。
   `token_contributions` は `wa.n_tokens` を寄与の分母に使う。
3. **tokenizer の実挙動**：このモデルは日本語を「む/か/し」と1文字級に細かく割り、文単位の
   意味類似も弱い。テストは意味的大小に依存させず、**恒等性**（同一テキストの自己対応 score>0.99）と
   **構造**（厳密分解の差は特殊トークン分のみ・許容0.05／混入なし）で検証する。共通語の検証には
   tokenizer が1トークンにまとめる「太郎」を使う。
4. **キャッシュの罠**：ファイル修正が反映されないときは `sys.modules` キャッシュと `__pycache__`/
   `.pyc` を疑う。`rm -rf */__pycache__ .pytest_cache` ＋ `python -B` で確実に最新を実行。
   `inspect.getsource(module.func)` で「python が実際に実行しているソース」を直接確認できる。
