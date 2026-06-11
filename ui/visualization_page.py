"""Vector visualization page for the RAG judicial system.

Shows the embedding space behind the RAG pipeline: every chunk of
constitutional / criminal / civil / administrative law and every precedent
is a 768-dim vector, and this page makes that geometry visible.

Sections:

1. **書庫マップ (Archive Map)** — interactive 2D/3D scatter of every
   chunk, colored by document type and case category.  PCA or t-SNE
   reduction.  Lets the user see clusters at a glance.
2. **クエリ近傍探索 (Query Neighborhood)** — type a query, watch where
   it lands and which chunks light up around it.
3. **文書間の類似度ヒートマップ (Similarity Heatmap)** — average pairwise
   cosine similarity between the major document groups.  Quantifies how
   much overlap there is between e.g. 刑法 and 刑事判例.
4. **二文書比較ラボ (Two-Document Comparison)** — pick any two texts
   and inspect their cosine similarity.  Includes the *桃太郎 vs 浦島
   太郎 vs 三匹の子豚* demo so the geometry concept clicks.
5. **物語ベクトルラボ (Story Vector Lab)** — full demo with several
   classic folktales, showing how RAG sees "similar but not identical"
   stories.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rag_system.vector_analysis import (
    cosine_similarity,
    cosine_similarity_matrix,
    embed_texts,
    fetch_all_embeddings,
    get_collection_stats,
    group_label,
    kmeans_clusters,
    project_cached,
    project_to_2d,
    project_to_3d,
    short_preview,
    top_k_similar,
)
from ui.highlight_view import render_similarity_highlight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------

GROUP_COLORS: dict[str, str] = {
    "憲法": "#1f77b4",          # blue
    "刑法": "#d62728",          # red
    "民法": "#2ca02c",          # green
    "行政法": "#9467bd",        # purple
    "文化規制": "#ff7f0e",       # orange
    "倫理指針": "#17becf",       # cyan
    "判例 (刑事)": "#e377c2",   # pink
    "判例 (民事)": "#8c564b",   # brown
    "判例 (憲法)": "#7f7f7f",   # gray
    "その他": "#bcbd22",         # olive
}


# ---------------------------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_corpus() -> dict[str, Any]:
    """Pull every chunk + embedding from ChromaDB (cached per Streamlit run)."""
    data = fetch_all_embeddings()
    labels = [group_label(m) for m in data["metadatas"]]
    previews = [short_preview(d, max_chars=140) for d in data["documents"]]
    return {
        "ids": data["ids"],
        "embeddings": data["embeddings"],
        "documents": data["documents"],
        "metadatas": data["metadatas"],
        "labels": labels,
        "previews": previews,
    }


@st.cache_data(show_spinner=False)
def _stats() -> dict[str, Any]:
    return get_collection_stats()


@st.cache_data(show_spinner="埋め込みベクトルを計算中…")
def _embed_cached(texts: tuple[str, ...]) -> np.ndarray:
    """Cached wrapper around :func:`embed_texts` (tuples are hashable)."""
    return embed_texts(list(texts))


@st.cache_data(show_spinner="次元削減を実行中…")
def _project_2d_cached(
    embeddings_bytes: bytes,
    shape: tuple[int, int],
    method: str,
) -> np.ndarray:
    embeddings = np.frombuffer(embeddings_bytes, dtype=np.float32).reshape(shape)
    return project_cached(embeddings, n_components=2, method=method)


@st.cache_data(show_spinner="3D 次元削減を実行中…")
def _project_3d_cached(
    embeddings_bytes: bytes,
    shape: tuple[int, int],
    method: str,
) -> np.ndarray:
    embeddings = np.frombuffer(embeddings_bytes, dtype=np.float32).reshape(shape)
    return project_cached(embeddings, n_components=3, method=method)


def _project_2d(embeddings: np.ndarray, method: str) -> np.ndarray:
    contig = np.ascontiguousarray(embeddings, dtype=np.float32)
    return _project_2d_cached(contig.tobytes(), contig.shape, method)


def _project_3d(embeddings: np.ndarray, method: str) -> np.ndarray:
    contig = np.ascontiguousarray(embeddings, dtype=np.float32)
    return _project_3d_cached(contig.tobytes(), contig.shape, method)


# ---------------------------------------------------------------------------
# Section 0: RAG Flow Walkthrough  (the "what is RAG actually doing?" view)
# ---------------------------------------------------------------------------


_RAG_FLOW_PRESETS: list[str] = [
    "窃盗罪で再犯の場合の量刑は？",
    "正当防衛の成立要件を述べよ",
    "未成年者の刑事責任",
    "契約の取消と無効の違い",
]


def _check_ollama_available() -> bool:
    """Best-effort ping of the Ollama server (no exceptions thrown)."""
    import urllib.error
    import urllib.request

    from rag_system.config import OLLAMA_BASE_URL

    try:
        req = urllib.request.Request(OLLAMA_BASE_URL, method="GET")
        with urllib.request.urlopen(req, timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _render_rag_flow(corpus: dict[str, Any]) -> None:
    st.subheader("0. RAG フロー可視化 ― 質問が回答になるまでの全工程")
    st.caption(
        "「RAG（Retrieval-Augmented Generation）」は次の 4 段階で動きます。"
        "ベクトル可視化ページの他のセクションは主に **②検索** を見ているだけですが、"
        "ここでは **①埋め込み → ②検索 → ③プロンプト構築 → ④生成** を順を追って見えるようにします。"
    )

    with st.expander("📘 RAG の基本 ― なぜこれが必要？", expanded=False):
        st.markdown(
            """
            #### RAG の正式名は **Retrieval-Augmented Generation**

            日本語訳すると「**検索で補強した生成**」。LLM 単体だと困る次の問題を解決します：

            | LLM 単体の弱点 | RAG が解決する仕組み |
            |---|---|
            | 学習時にない最新知識を知らない | **書庫から検索** して都度プロンプトに混ぜる |
            | 自信満々に嘘をつく（ハルシネーション） | **実在する文書だけ** を根拠に答えさせる |
            | ドメイン固有知識（例：謎の国家の刑法）が薄い | 専門書庫を **ベクトル DB** に詰めて引かせる |

            #### 4 つのステップ

            ```
            ①埋め込み      ②検索           ③プロンプト構築    ④生成
            ──────────    ──────────       ─────────────     ──────────
            質問テキスト → 質問ベクトル → 上位 K 件文書 → 大きな文章 → LLM 回答
            (768 次元)    (1354 件と比較)  (条文+判例)     (テンプレート)
            ```

            - **①埋め込み** は `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
              が担当（同じモデルでチャンク取り込み時もエンコードしているので、
              質問と書庫が同じ空間に乗る）。
            - **②検索** はチャンクと質問の **コサイン類似度** を計算して上位 K 件を返す
              （このページの他のセクションで散々見ている部分）。
            - **③プロンプト構築** は取得チャンクを `{context}` に、質問を `{question}` に
              埋め込んだ **長いプロンプト** を作る。LangChain の "stuff" chain と呼ばれる方式。
            - **④生成** はそのプロンプトを Ollama に投げて、回答テキストを得る。
              プロンプトに条文番号・判例 ID が入っているので、LLM は引用付きで答えられる。

            **このプロジェクトの実装は**：
            `rag_system/judge.py` の `JUDICIAL_PROMPT_TEMPLATE` を見ると、
            上の流れがそのままコードになっているのが確認できます。
            """
        )

    # --- Query input ---
    preset = st.selectbox(
        "プリセット質問",
        options=["(自由入力)"] + _RAG_FLOW_PRESETS,
        index=1,
        key="viz_rag_flow_preset",
    )
    default_q = preset if preset != "(自由入力)" else ""
    query = st.text_input(
        "質問",
        value=default_q,
        key=f"viz_rag_flow_query_{preset}",
    )

    col_k, col_run = st.columns([1, 1])
    with col_k:
        k = st.slider(
            "検索件数 K",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="viz_rag_flow_k",
        )
    with col_run:
        ollama_ok = _check_ollama_available()
        run_llm = st.checkbox(
            "④生成まで実行（Ollama 必須）",
            value=False,
            key="viz_rag_flow_run_llm",
            disabled=not ollama_ok,
            help=(
                "オフの場合は ③ プロンプト構築までを表示します。"
                "Ollama サーバーが起動していないとオンにできません。"
            ),
        )
        if not ollama_ok:
            st.caption("⚠ Ollama 未接続。①②③ までは動きます。")

    if not query.strip():
        st.info("質問を入力してください。")
        return

    st.markdown("---")

    # --- ① Embedding ---
    st.markdown("### ① 質問を埋め込む（テキスト → 768 次元ベクトル）")
    q_emb = _embed_cached((query.strip(),))[0]
    norm = float(np.linalg.norm(q_emb))

    col_a, col_b = st.columns([2, 1])
    with col_a:
        # Show first 32 dimensions as a tiny bar chart
        first_dims = q_emb[:32]
        fig = go.Figure(go.Bar(
            x=list(range(32)),
            y=first_dims,
            marker={"color": ["#1f77b4" if v >= 0 else "#d62728" for v in first_dims]},
        ))
        fig.update_layout(
            height=220,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            title="先頭 32 次元のベクトル値（全 768 次元の一部）",
            xaxis_title="次元 index",
            yaxis_title="値",
        )
        st.plotly_chart(fig, width="stretch")
    with col_b:
        st.metric("ベクトル次元", "768")
        st.metric("L2 ノルム", f"{norm:.3f}")
        st.metric("最大成分", f"{float(q_emb.max()):.3f}")
        st.metric("最小成分", f"{float(q_emb.min()):.3f}")
    st.caption(
        "▶ 文字列がそのまま 768 個の浮動小数になります。"
        "意味の近い質問は近いベクトルに、無関係な質問は遠いベクトルになります。"
    )

    # --- ② Vector search ---
    st.markdown("### ② 書庫から近傍 K 件を取り出す（コサイン類似度）")
    idx, scores = top_k_similar(q_emb, corpus["embeddings"], k=k)

    rows = []
    for rank, (i, score) in enumerate(zip(idx, scores), 1):
        meta = corpus["metadatas"][i]
        rows.append({
            "順位": rank,
            "類似度": float(score),
            "グループ": corpus["labels"][i],
            "Source": meta.get("source", ""),
            "Case ID": meta.get("case_id", ""),
        })
    df_search = pd.DataFrame(rows)
    st.dataframe(
        df_search,
        width="stretch",
        hide_index=True,
        column_config={
            "類似度": st.column_config.ProgressColumn(
                "類似度", format="%.3f", min_value=0.0, max_value=1.0
            ),
        },
    )
    st.caption(
        "▶ 1,354 チャンク全件と質問ベクトルのコサイン類似度を計算し、"
        "上位 K 件を抜き出します。これが「RAG の R」の本体。"
        "他のセクションで深掘りしているのもこの部分です。"
    )

    # --- ③ Prompt construction ---
    st.markdown("### ③ プロンプトを組み立てる（取得文書 + 質問 → 大きな指示文）")
    st.caption(
        "LangChain の \"stuff\" chain は、取得した K 件の文書を全部つなげて "
        "プロンプトテンプレートの `{context}` に流し込み、"
        "質問を `{question}` に流し込みます。"
    )

    # Build the same context string that judge.py would
    context_chunks = []
    for rank, i in enumerate(idx, 1):
        meta = corpus["metadatas"][i]
        head = f"[資料{rank}] {corpus['labels'][i]}"
        if meta.get("case_id"):
            head += f" — {meta['case_id']}"
        if meta.get("source"):
            head += f" ({meta['source']})"
        body = corpus["documents"][i]
        # Keep body short for display purposes
        if len(body) > 600:
            body = body[:600] + "…"
        context_chunks.append(f"{head}\n{body}")
    context_text = "\n\n".join(context_chunks)

    from rag_system.judge import JUDICIAL_PROMPT_TEMPLATE

    full_prompt = JUDICIAL_PROMPT_TEMPLATE.format(
        context=context_text, question=query.strip()
    )

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("**プロンプトテンプレート（雛形）**")
        st.code(JUDICIAL_PROMPT_TEMPLATE[:600] + "…", language="text")
    with col_r:
        st.markdown(
            f"**実際に LLM に渡される全文**（{len(full_prompt):,} 文字 / "
            f"{len(full_prompt.encode('utf-8')):,} bytes）"
        )
        with st.expander("クリックして全文を見る", expanded=False):
            st.text(full_prompt)

    st.caption(
        "▶ 取得文書を `{context}` に、質問を `{question}` に差し込んだだけ。"
        "LLM はこの長文プロンプトを丸ごと読んで、条文番号・判例 ID を "
        "引用しながら回答するように指示されています。"
        "これが「RAG の A（Augmented）」、つまり「検索結果でプロンプトを補強する」工程です。"
    )

    # --- ④ Generation ---
    st.markdown("### ④ LLM に生成させる（プロンプト → 回答テキスト）")

    if not run_llm:
        if ollama_ok:
            st.info(
                "ここで上のプロンプトを Ollama LLM に送り、回答を受け取ります。"
                "「④生成まで実行」にチェックを入れて再実行すると実際に生成されます。"
            )
        else:
            st.warning(
                "Ollama サーバーが起動していないため、生成は実行できません。"
                "`ollama serve` で起動してから再読み込みしてください。"
            )
        return

    with st.spinner("Ollama が回答を生成中…"):
        try:
            from rag_system.judge import create_llm

            llm = create_llm()
            answer = llm.invoke(full_prompt)
        except ConnectionError:
            st.error(
                "Ollama サーバーへの接続に失敗しました。"
                "`ollama serve` を実行してから再試行してください。"
            )
            return
        except Exception as exc:  # pragma: no cover - depends on env
            st.error(f"生成中にエラーが発生しました: {exc}")
            return

    st.markdown("**生成された回答**")
    st.markdown(answer)
    st.caption(
        "▶ ここまでが RAG の全工程です。"
        "回答の中に条文番号や判例 ID が現れるのは、③で渡したプロンプトに"
        "それらの情報が入っているから。"
        "もし回答が薄ければ ② で取得した文書が論点とズレている可能性が高い。"
    )


# ---------------------------------------------------------------------------
# Section 1: Archive Map
# ---------------------------------------------------------------------------


def _render_archive_map(corpus: dict[str, Any]) -> None:
    st.subheader("1. 書庫マップ ― 全文書のベクトル空間")
    st.caption(
        "全 1,354 チャンクを 768 次元 → 2D/3D に圧縮して描画。"
        "近い点は似た意味、遠い点は意味が違うことを示します。"
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        dims = st.radio(
            "次元",
            options=["2D", "3D"],
            index=0,
            horizontal=True,
            key="viz_archive_dims",
        )
    with col2:
        method = st.radio(
            "削減手法",
            options=["pca", "tsne"],
            index=0,
            horizontal=True,
            key="viz_archive_method",
            format_func=lambda x: x.upper(),
            help=(
                "PCA は線形・高速・全体構造を保持。"
                "t-SNE は非線形・低速だが局所クラスタが鮮明。"
            ),
        )
    with col3:
        available_groups = sorted({lbl for lbl in corpus["labels"]})
        selected_groups = st.multiselect(
            "表示するグループ",
            options=available_groups,
            default=available_groups,
            key="viz_archive_groups",
        )

    if not selected_groups:
        st.info("少なくとも一つのグループを選択してください。")
        return

    mask = np.array(
        [lbl in selected_groups for lbl in corpus["labels"]], dtype=bool
    )
    if not mask.any():
        st.info("選択条件に該当する文書がありません。")
        return

    embeddings = corpus["embeddings"][mask]
    labels = [lbl for lbl, keep in zip(corpus["labels"], mask) if keep]
    previews = [p for p, keep in zip(corpus["previews"], mask) if keep]
    metadatas = [m for m, keep in zip(corpus["metadatas"], mask) if keep]

    sources = [m.get("source", "") for m in metadatas]
    case_ids = [m.get("case_id", "") for m in metadatas]

    if dims == "2D":
        coords = _project_2d(embeddings, method=method)
        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "group": labels,
            "source": sources,
            "case_id": case_ids,
            "preview": previews,
        })
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="group",
            color_discrete_map=GROUP_COLORS,
            hover_data={"source": True, "case_id": True, "preview": True,
                        "x": False, "y": False, "group": True},
            opacity=0.7,
        )
        fig.update_traces(marker={"size": 6})
        fig.update_layout(
            height=600,
            xaxis_title=f"{method.upper()} 第1主成分",
            yaxis_title=f"{method.upper()} 第2主成分",
            legend_title="グループ",
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
        )
        st.plotly_chart(fig, width="stretch")
    else:
        coords = _project_3d(embeddings, method=method)
        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "group": labels,
            "source": sources,
            "case_id": case_ids,
            "preview": previews,
        })
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="group",
            color_discrete_map=GROUP_COLORS,
            hover_data={"source": True, "case_id": True, "preview": True,
                        "x": False, "y": False, "z": False, "group": True},
            opacity=0.7,
        )
        fig.update_traces(marker={"size": 3})
        fig.update_layout(
            height=700,
            scene={
                "xaxis_title": f"{method.upper()}-1",
                "yaxis_title": f"{method.upper()}-2",
                "zaxis_title": f"{method.upper()}-3",
            },
            legend_title="グループ",
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
        )
        st.plotly_chart(fig, width="stretch")

    with st.expander("グループ別の点数"):
        counts = pd.Series(labels).value_counts().rename_axis("グループ").reset_index(name="点数")
        st.dataframe(counts, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Section 2: Query Neighborhood
# ---------------------------------------------------------------------------


_QUERY_PRESETS: list[str] = [
    "窃盗罪で再犯の場合の量刑は？",
    "正当防衛と過剰防衛の境界はどこにある？",
    "民事における精神的損害賠償の認定基準",
    "表現の自由とプライバシー権の調整",
    "未成年者の刑事責任",
    "契約の無効・取消・解除の違い",
    "行政処分に対する不服申立の手続",
    "AI 生成物の著作権はどう扱われるか",
]


def _render_query_neighborhood(corpus: dict[str, Any]) -> None:
    st.subheader("2. クエリ近傍探索 ― 質問はベクトル空間のどこに落ちる？")
    st.caption(
        "入力した質問を 768 次元に埋め込み、書庫上の近い順に表示。"
        "右の散布図ではクエリ位置（★）と近傍点（■）が強調されます。"
    )

    # Preset query chips
    preset_label = st.selectbox(
        "プリセット質問（選ぶと下のテキストが置き換わります）",
        options=["(自由入力)"] + _QUERY_PRESETS,
        index=0,
        key="viz_query_preset",
    )

    col_input, col_settings = st.columns([3, 1])
    with col_input:
        default_query = (
            preset_label
            if preset_label != "(自由入力)"
            else st.session_state.get(
                "viz_query_text", "窃盗罪で再犯の場合の量刑は？"
            )
        )
        query = st.text_input(
            "クエリ",
            value=default_query,
            key=f"viz_query_text_{preset_label}",
        )
    with col_settings:
        k = st.slider("近傍数 K", min_value=3, max_value=30,
                      value=10, step=1, key="viz_query_k")

    if not query.strip():
        st.info("クエリを入力してください。")
        return

    query_emb = _embed_cached((query.strip(),))[0]

    idx, scores = top_k_similar(query_emb, corpus["embeddings"], k=k)

    # --- All-corpus similarity for distribution plot ---
    all_sims = cosine_similarity_matrix(
        query_emb[None, :], corpus["embeddings"]
    )[0]

    # Headline metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最大類似度", f"{float(all_sims.max()):.3f}")
    m2.metric("平均類似度", f"{float(all_sims.mean()):.3f}")
    m3.metric("中央値", f"{float(np.median(all_sims)):.3f}")
    m4.metric("最小類似度", f"{float(all_sims.min()):.3f}")

    # --- Two columns: top-K list (left) and projection (right) ---
    col_list, col_plot = st.columns([1, 1])

    with col_list:
        st.markdown("**近傍 K 件 (類似度の高い順)**")
        rows = []
        for rank, (i, score) in enumerate(zip(idx, scores), 1):
            meta = corpus["metadatas"][i]
            rows.append({
                "順位": rank,
                "類似度": float(score),
                "グループ": corpus["labels"][i],
                "Source": meta.get("source", ""),
                "Case ID": meta.get("case_id", ""),
            })
        df_top = pd.DataFrame(rows)
        st.dataframe(
            df_top,
            width="stretch",
            hide_index=True,
            column_config={
                "類似度": st.column_config.ProgressColumn(
                    "類似度",
                    format="%.3f",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
        )
        with st.expander("最近傍の本文プレビュー"):
            for rank, (i, score) in enumerate(zip(idx, scores), 1):
                meta = corpus["metadatas"][i]
                head = f"#{rank} [{corpus['labels'][i]}] 類似度 {score:.3f}"
                source = meta.get("source", "")
                if source:
                    head += f" — {source}"
                case_id = meta.get("case_id", "")
                if case_id:
                    head += f" ({case_id})"
                st.markdown(f"**{head}**")
                preview = corpus["documents"][i]
                if len(preview) > 400:
                    preview = preview[:400] + "…"
                st.text(preview)
                if rank < len(idx):
                    st.divider()

    with col_plot:
        # Project query + corpus together so the query sits in the same plane
        sample_size = min(500, corpus["embeddings"].shape[0])
        if corpus["embeddings"].shape[0] > sample_size:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(
                corpus["embeddings"].shape[0], size=sample_size, replace=False
            )
        else:
            sample_idx = np.arange(corpus["embeddings"].shape[0])

        sample_emb = corpus["embeddings"][sample_idx]
        # Always include the top-K so they appear in the plot
        ensure_idx = np.array(idx)
        extra = np.setdiff1d(ensure_idx, sample_idx)
        if extra.size > 0:
            sample_idx = np.concatenate([sample_idx, extra])
            sample_emb = corpus["embeddings"][sample_idx]

        # Stack query + sample, project together
        combined = np.vstack([query_emb[None, :], sample_emb])
        coords = _project_2d(combined, method="pca")
        q_xy = coords[0]
        sample_xy = coords[1:]

        # Tag each sample point
        sample_labels = [corpus["labels"][i] for i in sample_idx]
        sample_in_topk = np.isin(sample_idx, ensure_idx)

        plot_df = pd.DataFrame({
            "x": sample_xy[:, 0],
            "y": sample_xy[:, 1],
            "group": sample_labels,
            "topk": ["近傍" if v else "その他" for v in sample_in_topk],
        })

        fig = go.Figure()

        # Background: other points (faded)
        bg = plot_df[plot_df["topk"] == "その他"]
        fig.add_trace(go.Scatter(
            x=bg["x"], y=bg["y"], mode="markers",
            marker={"size": 4, "color": "lightgray", "opacity": 0.4},
            name="書庫の他文書",
            hovertext=bg["group"],
            hoverinfo="text",
        ))

        # Top-K neighbors (highlighted by group color)
        topk = plot_df[plot_df["topk"] == "近傍"]
        for grp, color in GROUP_COLORS.items():
            sub = topk[topk["group"] == grp]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"], mode="markers",
                marker={"size": 12, "color": color,
                        "line": {"width": 1, "color": "black"}},
                name=f"近傍: {grp}",
                hovertext=sub["group"],
                hoverinfo="text",
            ))

        # Query point (star)
        fig.add_trace(go.Scatter(
            x=[q_xy[0]], y=[q_xy[1]], mode="markers+text",
            marker={"size": 22, "color": "gold", "symbol": "star",
                    "line": {"width": 2, "color": "black"}},
            text=["クエリ"],
            textposition="top center",
            name="クエリ",
            hovertext=[query],
            hoverinfo="text",
        ))

        fig.update_layout(
            title="PCA 2D 射影上のクエリ位置",
            height=560,
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            xaxis_title="PC1",
            yaxis_title="PC2",
        )
        st.plotly_chart(fig, width="stretch")

    # --- Distribution histogram ---
    st.markdown("**書庫全体に対する類似度分布**")
    hist_df = pd.DataFrame({
        "類似度": all_sims,
        "グループ": corpus["labels"],
    })
    fig_hist = px.histogram(
        hist_df,
        x="類似度",
        color="グループ",
        color_discrete_map=GROUP_COLORS,
        nbins=50,
        opacity=0.75,
        barmode="stack",
    )
    fig_hist.add_vline(
        x=float(all_sims.mean()),
        line_dash="dash",
        line_color="black",
        annotation_text=f"平均 {float(all_sims.mean()):.3f}",
        annotation_position="top right",
    )
    fig_hist.update_layout(
        height=320,
        bargap=0.05,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(fig_hist, width="stretch")
    st.caption(
        "右側（高類似度）にどのグループが多いかで、クエリが「どの法律分野に属するか」が読み取れます。"
        "左右にうまく裾を引いているクエリは書庫の特定領域にヒットしている良いクエリ、"
        "ピークが中央に寄っていると曖昧なクエリです。"
    )


# ---------------------------------------------------------------------------
# Section 2.5: K-means Cluster Analysis
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="クラスタリングを実行中…")
def _kmeans_cached(
    embeddings_bytes: bytes,
    shape: tuple[int, int],
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.frombuffer(embeddings_bytes, dtype=np.float32).reshape(shape)
    return kmeans_clusters(embeddings, n_clusters=n_clusters)


def _render_cluster_analysis(corpus: dict[str, Any]) -> None:
    st.subheader("3. クラスタ自動発見 ― K-means で書庫を群分けする")
    st.caption(
        "メタデータ（憲法／刑法／判例…）を一切見ずに、ベクトルの近さだけで"
        "K-means が自動的にグループ分けします。"
        "人手のラベルと自動クラスタが一致するなら RAG の意味理解は健全、"
        "ズレるなら「同じカテゴリでも内容はかなり違う」サブトピックが眠っています。"
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        n_clusters = st.slider(
            "クラスタ数 K",
            min_value=2,
            max_value=20,
            value=8,
            step=1,
            key="viz_cluster_k",
        )
        method = st.radio(
            "可視化手法",
            options=["pca", "tsne"],
            index=0,
            key="viz_cluster_method",
            format_func=lambda x: x.upper(),
            horizontal=True,
        )

    embeddings = corpus["embeddings"]
    contig = np.ascontiguousarray(embeddings, dtype=np.float32)
    labels_int, _ = _kmeans_cached(contig.tobytes(), contig.shape, n_clusters)
    coords = _project_2d(embeddings, method=method)

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": [f"クラスタ {c}" for c in labels_int],
        "human_label": corpus["labels"],
        "source": [m.get("source", "") for m in corpus["metadatas"]],
        "preview": corpus["previews"],
    })

    with col2:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            hover_data={
                "human_label": True,
                "source": True,
                "preview": True,
                "x": False,
                "y": False,
                "cluster": True,
            },
            opacity=0.7,
        )
        fig.update_traces(marker={"size": 5})
        fig.update_layout(
            height=520,
            xaxis_title=f"{method.upper()}-1",
            yaxis_title=f"{method.upper()}-2",
            legend_title="自動クラスタ",
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
        )
        st.plotly_chart(fig, width="stretch")

    # Cross-tab: human labels vs cluster labels
    st.markdown("**クラスタ × 人手ラベル のクロス集計**")
    ct = pd.crosstab(df["human_label"], df["cluster"]).rename_axis(
        index="人手ラベル", columns="自動クラスタ"
    )
    st.dataframe(ct, width="stretch")
    st.caption(
        "行（人手ラベル）が一つのクラスタに集中していれば、その分類は意味的にまとまっています。"
        "複数クラスタに分散していれば、その分類内でも話題が複数あるという信号です。"
    )


# ---------------------------------------------------------------------------
# Section 3: Group Similarity Heatmap
# ---------------------------------------------------------------------------


def _render_group_heatmap(corpus: dict[str, Any]) -> None:
    st.subheader("4. グループ間類似度ヒートマップ")
    st.caption(
        "各グループの重心ベクトル同士のコサイン類似度。"
        "「刑法と刑事判例」のように密接な関係は赤、無関係は青になります。"
    )

    labels = corpus["labels"]
    embeddings = corpus["embeddings"]

    unique_groups = sorted(set(labels))
    centroids = []
    counts = []
    for grp in unique_groups:
        mask = np.array([lbl == grp for lbl in labels], dtype=bool)
        sub = embeddings[mask]
        if sub.shape[0] == 0:
            centroids.append(np.zeros(embeddings.shape[1], dtype=np.float32))
            counts.append(0)
            continue
        centroids.append(sub.mean(axis=0))
        counts.append(int(sub.shape[0]))
    centroid_matrix = np.stack(centroids, axis=0)

    sim = cosine_similarity_matrix(centroid_matrix)

    fig = go.Figure(go.Heatmap(
        z=sim,
        x=unique_groups,
        y=unique_groups,
        colorscale="RdBu_r",
        zmin=0.0,
        zmax=1.0,
        text=[[f"{v:.2f}" for v in row] for row in sim],
        texttemplate="%{text}",
        textfont={"size": 11},
        hoverongaps=False,
    ))
    fig.update_layout(
        height=540,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(fig, width="stretch")

    # CSV download
    sim_df = pd.DataFrame(sim, index=unique_groups, columns=unique_groups)
    csv_bytes = sim_df.to_csv().encode("utf-8")
    st.download_button(
        label="📥 類似度マトリックスを CSV でダウンロード",
        data=csv_bytes,
        file_name="group_similarity_matrix.csv",
        mime="text/csv",
    )

    with st.expander("各グループの点数（凡例）"):
        df = pd.DataFrame({"グループ": unique_groups, "点数": counts})
        st.dataframe(df.sort_values("点数", ascending=False),
                     width="stretch", hide_index=True)

    # --- Verdict comparison (有罪 vs 無罪) ---
    _render_verdict_comparison(corpus)


def _render_verdict_comparison(corpus: dict[str, Any]) -> None:
    """Subsection: average vector of 有罪 vs 無罪 judgments."""
    metadatas = corpus["metadatas"]
    embeddings = corpus["embeddings"]

    verdicts = [m.get("verdict", "") for m in metadatas]
    unique_verdicts = sorted({v for v in verdicts if v})
    if not unique_verdicts:
        return

    st.markdown("**4-a. 判決種別ごとの重心比較**")
    st.caption(
        "判例だけを対象に、判決（有罪・無罪・一部認容…）ごとの重心ベクトルを比較。"
        "判決の違いが文書ベクトルにどれだけ現れるかを定量化します。"
    )

    centroids = []
    counts = []
    for v in unique_verdicts:
        mask = np.array([vv == v for vv in verdicts], dtype=bool)
        sub = embeddings[mask]
        if sub.shape[0] == 0:
            continue
        centroids.append(sub.mean(axis=0))
        counts.append(int(sub.shape[0]))

    centroid_matrix = np.stack(centroids, axis=0)
    sim = cosine_similarity_matrix(centroid_matrix)

    fig = go.Figure(go.Heatmap(
        z=sim,
        x=unique_verdicts,
        y=unique_verdicts,
        colorscale="RdBu_r",
        zmin=0.5,
        zmax=1.0,
        text=[[f"{v:.3f}" for v in row] for row in sim],
        texttemplate="%{text}",
        textfont={"size": 11},
    ))
    fig.update_layout(
        height=360,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(fig, width="stretch")

    # CSV download
    sim_df = pd.DataFrame(sim, index=unique_verdicts, columns=unique_verdicts)
    csv_bytes = sim_df.to_csv().encode("utf-8")
    st.download_button(
        label="📥 判決類似度マトリックスを CSV でダウンロード",
        data=csv_bytes,
        file_name="verdict_similarity_matrix.csv",
        mime="text/csv",
    )

    with st.expander("判決ごとの判例数"):
        df = pd.DataFrame({"判決": unique_verdicts, "判例数": counts})
        st.dataframe(
            df.sort_values("判例数", ascending=False),
            width="stretch", hide_index=True,
        )
    st.caption(
        "判決同士の類似度が高い（例：「有罪」と「一部有罪」が 0.95 以上）と、"
        "ベクトル空間では結果が違っても扱う問題は似ている、"
        "つまり「論点は同じだが結論が違うケース」が多いことを意味します。"
    )


# ---------------------------------------------------------------------------
# Section 4.5: Neighbor Explorer (pick a chunk, find its neighbors)
# ---------------------------------------------------------------------------


def _render_neighbor_explorer(corpus: dict[str, Any]) -> None:
    st.subheader("5. 文書近傍エクスプローラ ― ある文書に最も似ている K 件を見る")
    st.caption(
        "コーパスから一つチャンクを選ぶと、そのベクトルに最も近い文書を表示します。"
        "「この窃盗判例に近い別の窃盗判例はあるか？」「刑法のこの条文に呼応する判例はどれか？」"
        "を直接たどれます。"
    )

    embeddings = corpus["embeddings"]
    metadatas = corpus["metadatas"]
    documents = corpus["documents"]
    labels = corpus["labels"]

    # Filter by group first to make selection manageable
    available_groups = sorted(set(labels))
    selected_group = st.selectbox(
        "起点とするグループ",
        options=available_groups,
        index=0,
        key="viz_neighbor_group",
    )

    mask = np.array([lbl == selected_group for lbl in labels], dtype=bool)
    idx_in_group = np.where(mask)[0]
    if idx_in_group.size == 0:
        st.info("該当する文書がありません。")
        return

    # Build display labels
    options = []
    for i in idx_in_group[:300]:  # cap at 300 to keep selectbox snappy
        meta = metadatas[i]
        title = meta.get("case_id") or meta.get("source", "")[-40:]
        preview = short_preview(documents[i], 40)
        options.append((i, f"{title} — {preview}"))

    selected_label = st.selectbox(
        "起点の文書",
        options=[label for _, label in options],
        index=0,
        key="viz_neighbor_source",
    )
    selected_idx = next(
        i for i, label in options if label == selected_label
    )

    k = st.slider("近傍数 K", min_value=3, max_value=20, value=8,
                  step=1, key="viz_neighbor_k")

    query_emb = embeddings[selected_idx]

    # Exclude the point itself
    sims = cosine_similarity_matrix(
        query_emb[None, :], embeddings
    )[0]
    sims[selected_idx] = -np.inf
    top_idx = np.argsort(-sims)[:k]
    top_scores = sims[top_idx]

    # --- Source view ---
    with st.expander("起点の本文を見る", expanded=False):
        st.markdown(f"**{labels[selected_idx]}**")
        st.text(documents[selected_idx][:1500] + (
            "…" if len(documents[selected_idx]) > 1500 else ""
        ))

    # --- Top-K table ---
    rows = []
    for rank, (i, score) in enumerate(zip(top_idx, top_scores), 1):
        rows.append({
            "順位": rank,
            "類似度": float(score),
            "グループ": labels[i],
            "Source": metadatas[i].get("source", ""),
            "Case ID": metadatas[i].get("case_id", ""),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "類似度": st.column_config.ProgressColumn(
                "類似度", format="%.3f", min_value=0.0, max_value=1.0
            ),
        },
    )

    with st.expander("近傍文書の本文プレビュー"):
        for rank, (i, score) in enumerate(zip(top_idx, top_scores), 1):
            meta = metadatas[i]
            head = (
                f"#{rank} [{labels[i]}] 類似度 {score:.3f}"
            )
            if meta.get("source"):
                head += f" — {meta['source']}"
            if meta.get("case_id"):
                head += f" ({meta['case_id']})"
            st.markdown(f"**{head}**")
            preview = documents[i]
            if len(preview) > 400:
                preview = preview[:400] + "…"
            st.text(preview)
            if rank < len(top_idx):
                st.divider()


# ---------------------------------------------------------------------------
# Section 4: Two-Document Comparison
# ---------------------------------------------------------------------------


_TEXT_PRESETS: dict[str, str] = {
    "(自由入力)": "",
    "桃太郎": (
        "むかしむかし、ある所におじいさんとおばあさんがいた。"
        "おばあさんが川で洗濯をしていると大きな桃が流れてきた。"
        "桃を割ると元気な男の子が現れた。男の子は桃太郎と名付けられ、"
        "やがて鬼ヶ島へ鬼退治の旅に出た。イヌ・サル・キジを家来にして"
        "鬼を倒し、宝物を持ち帰った。"
    ),
    "浦島太郎": (
        "むかしむかし、浦島太郎という若い漁師がいた。"
        "ある日、子供たちにいじめられていた亀を助けた。"
        "亀のお礼に竜宮城へ連れて行かれ、乙姫様と楽しく過ごした。"
        "故郷に戻ると何百年もの時が経っており、玉手箱を開けると"
        "白い煙が出て老人になってしまった。"
    ),
    "三匹の子豚": (
        "三匹の子豚兄弟が独立して家を建てた。"
        "一番目はわらの家、二番目は木の家、三番目はレンガの家。"
        "オオカミがやってきて、わらの家と木の家を吹き飛ばしたが、"
        "レンガの家だけは壊せなかった。最後にオオカミは煙突から侵入"
        "しようとして煮えたお湯に落ちて退治された。"
    ),
    "シンデレラ": (
        "継母と義姉にいじめられていたシンデレラ。"
        "魔法使いに助けられ、舞踏会へ行き王子と出会う。"
        "12時の鐘で慌てて帰る途中、ガラスの靴を片方落としてしまう。"
        "王子はその靴を頼りに国中を探し回り、ぴったりだったシンデレラと結婚した。"
    ),
    "謎の国家 — 窃盗罪の量刑": (
        "謎の国家の刑法第235条によれば、他人の財物を窃取した者は窃盗罪に"
        "問われる。量刑は10年以下の懲役または50万円以下の罰金とし、"
        "被害金額、態様、再犯の有無、被害弁償の状況を考慮して定める。"
    ),
    "謎の国家 — 不法行為": (
        "謎の国家の民法第709条によれば、故意又は過失によって他人の権利"
        "又は法律上保護される利益を侵害した者は、これによって生じた損害を"
        "賠償する責任を負う。損害賠償は財産的損害と精神的損害を含む。"
    ),
    "謎の国家 — 表現の自由": (
        "謎の国家憲法第21条は、集会、結社及び言論、出版その他一切の表現の"
        "自由を保障する。公共の福祉に反しない限り最大限尊重されるが、"
        "他者の名誉やプライバシーとの調整が必要である。"
    ),
}


def _render_two_doc_comparison() -> None:
    st.subheader("6. 二文書比較ラボ ― RAG はどう「似ている」を測るか")
    st.caption(
        "二つのテキストの埋め込みベクトルを取り、コサイン類似度を計算します。"
        "0 に近いほど無関係、1 に近いほど似た意味。-1 は意味が真逆。"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**文書 A**")
        preset_a = st.selectbox(
            "プリセット",
            options=list(_TEXT_PRESETS.keys()),
            index=1,
            key="viz_compare_preset_a",
        )
        default_a = _TEXT_PRESETS[preset_a]
        text_a = st.text_area(
            "文書 A の本文",
            value=default_a,
            height=200,
            key=f"viz_compare_text_a_{preset_a}",
        )
    with col_b:
        st.markdown("**文書 B**")
        preset_b = st.selectbox(
            "プリセット",
            options=list(_TEXT_PRESETS.keys()),
            index=2,
            key="viz_compare_preset_b",
        )
        default_b = _TEXT_PRESETS[preset_b]
        text_b = st.text_area(
            "文書 B の本文",
            value=default_b,
            height=200,
            key=f"viz_compare_text_b_{preset_b}",
        )

    if not text_a.strip() or not text_b.strip():
        st.info("両方の文書に本文を入力してください。")
        return

    emb = _embed_cached((text_a.strip(), text_b.strip()))
    sim = cosine_similarity(emb[0], emb[1])

    # Headline metric
    col_metric, col_gauge = st.columns([1, 2])
    with col_metric:
        st.metric("コサイン類似度", f"{sim:.4f}")
        interpretation = _interpret_similarity(sim)
        st.write(interpretation)
    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sim,
            number={"valueformat": ".3f"},
            gauge={
                "axis": {"range": [-0.2, 1.0]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [-0.2, 0.3], "color": "#deebf7"},
                    {"range": [0.3, 0.5], "color": "#9ecae1"},
                    {"range": [0.5, 0.7], "color": "#6baed6"},
                    {"range": [0.7, 0.85], "color": "#3182bd"},
                    {"range": [0.85, 1.0], "color": "#08519c"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": sim,
                },
            },
        ))
        fig.update_layout(height=240,
                          margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(fig, width="stretch")

    # Component-wise breakdown (top contributing dimensions)
    with st.expander("ベクトルの中身を見る (768 次元の上位寄与)"):
        v1 = emb[0] / (np.linalg.norm(emb[0]) or 1.0)
        v2 = emb[1] / (np.linalg.norm(emb[1]) or 1.0)
        contrib = v1 * v2  # element-wise contribution to cosine sim
        order = np.argsort(-np.abs(contrib))[:32]
        contrib_df = pd.DataFrame({
            "次元": order,
            "A 成分": emb[0][order],
            "B 成分": emb[1][order],
            "寄与 (正)": contrib[order],
        }).sort_values("寄与 (正)", ascending=False)
        st.dataframe(
            contrib_df,
            width="stretch",
            hide_index=True,
            column_config={
                "A 成分": st.column_config.NumberColumn("A 成分", format="%.3f"),
                "B 成分": st.column_config.NumberColumn("B 成分", format="%.3f"),
                "寄与 (正)": st.column_config.NumberColumn(
                    "寄与 (正)", format="%.4f"
                ),
            },
        )
        st.caption(
            "「寄与 (正)」は正規化済みベクトルの要素積。"
            "総和がそのままコサイン類似度になります。"
            "正方向の大きい次元は両文書が共通して活性化している軸、"
            "負方向は片方だけが活性化している軸です。"
        )

    st.markdown("**どこが似ているか ― 実テキストのハイライト**")
    render_similarity_highlight(
        text_a,
        text_b,
        label_a=f"文書 A（{preset_a}）",
        label_b=f"文書 B（{preset_b}）",
        key_prefix="twodoc",
    )


def _interpret_similarity(sim: float) -> str:
    if sim >= 0.85:
        return "🟥 **非常に類似**：ほぼ同じ意味と判断される範囲。"
    if sim >= 0.70:
        return "🟧 **強く類似**：同じトピックの異なる表現。RAG は強く引き寄せる。"
    if sim >= 0.50:
        return "🟨 **やや類似**：共通の要素はあるが別物。RAG が混ぜがちな範囲。"
    if sim >= 0.30:
        return "🟦 **弱い類似**：ジャンルや雰囲気だけが共通。"
    return "⬜ **ほぼ無関係**：別の話題。"


# ---------------------------------------------------------------------------
# Section 5: Story Vector Lab (the 桃太郎 / 浦島太郎 / 三匹の子豚 demo)
# ---------------------------------------------------------------------------


_STORY_LAB: dict[str, str] = {
    "桃太郎": (
        "桃から生まれた男の子が、犬・猿・雉を従えて鬼ヶ島へ鬼退治に行く昔話。"
        "勧善懲悪、旅、仲間集め、宝物が主要モチーフ。"
    ),
    "浦島太郎": (
        "亀を助けた青年が竜宮城に招かれ、長い時を過ごした後に老人へと変わる昔話。"
        "異界訪問、時の流れ、玉手箱が主要モチーフ。"
    ),
    "金太郎": (
        "山で熊と相撲を取って育った力持ちの男の子の物語。"
        "怪力、動物の友、成長譚が中心。"
    ),
    "一寸法師": (
        "とても小さな男の子が京の都へ旅立ち、鬼を退治して大きくなる物語。"
        "小さい主人公、旅、鬼退治、立身が主要モチーフ。"
    ),
    "三匹の子豚": (
        "わら・木・レンガで家を建てた三兄弟の子豚と、それを襲うオオカミの物語。"
        "三つの選択肢、努力、強敵への防御が主要モチーフ。"
    ),
    "赤ずきん": (
        "おばあさんを訪ねる少女が森でオオカミに騙される童話。"
        "森、騙し、再生（猟師が助ける）が主要モチーフ。"
    ),
    "シンデレラ": (
        "継母にいじめられた娘が魔法で舞踏会へ行き、ガラスの靴で王子に見つけられる童話。"
        "苦難、魔法、舞踏会、靴が主要モチーフ。"
    ),
    "謎の国家・刑法": (
        "謎の国家の刑法は窃盗・強盗・殺人など犯罪の構成要件と量刑を定める。"
        "刑事責任、構成要件、違法性、有責性、量刑が主要概念。"
    ),
    "謎の国家・民法": (
        "謎の国家の民法は契約・不法行為・物権など私人間の法律関係を規律する。"
        "契約自由、損害賠償、所有権、相続が主要概念。"
    ),
    "謎の国家・憲法": (
        "謎の国家憲法は基本的人権・統治機構・法の支配を定める最高法規。"
        "表現の自由、平等、適正手続、三権分立が主要概念。"
    ),
}


def _render_story_lab() -> None:
    st.subheader("7. 物語ベクトルラボ ― 桃太郎と浦島太郎はどれくらい似ている？")
    st.caption(
        "RAG は文章を 768 次元の点として見ます。"
        "似た要素（旅・主人公・異界）が多いほど点同士は近付き、"
        "違うジャンルになると遠く離れます。"
        "ここでは古典童話と「謎の国家」の法令を同じ空間に並べて、"
        "RAG がどう「似ている／似ていない」を測るかを体感できます。"
    )

    # All texts
    titles = list(_STORY_LAB.keys())
    texts = list(_STORY_LAB.values())

    embeddings = _embed_cached(tuple(texts))

    # --- (a) similarity heatmap ---
    sim = cosine_similarity_matrix(embeddings)

    st.markdown("**7-a. 全ペアの類似度ヒートマップ**")
    fig_hm = go.Figure(go.Heatmap(
        z=sim,
        x=titles,
        y=titles,
        colorscale="RdBu_r",
        zmin=0.3,
        zmax=1.0,
        text=[[f"{v:.2f}" for v in row] for row in sim],
        texttemplate="%{text}",
        textfont={"size": 11},
    ))
    fig_hm.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(fig_hm, width="stretch")
    st.caption(
        "▶ 桃太郎・浦島太郎・一寸法師は「主人公が旅して何かを倒す／訪れる」"
        "という構造が共通するため類似度が高め。"
        "三匹の子豚や赤ずきんは別の物語類型なので少し離れる。"
        "謎の国家の法令はジャンルが全く違うので童話とは大きく離れて表示されます。"
    )

    # CSV download
    sim_df = pd.DataFrame(sim, index=titles, columns=titles)
    csv_bytes = sim_df.to_csv().encode("utf-8")
    st.download_button(
        label="📥 類似度マトリックスを CSV でダウンロード",
        data=csv_bytes,
        file_name="story_similarity_matrix.csv",
        mime="text/csv",
    )

    # --- (b) 2D + 3D projection toggle ---
    st.markdown("**7-b. 散布図 — 童話と法令は別クラスタになるか？**")
    view_dim = st.radio(
        "次元",
        options=["2D", "3D"],
        index=0,
        horizontal=True,
        key="viz_story_dim",
    )

    # Categorize
    def _category(name: str) -> str:
        if name.startswith("謎の国家"):
            return "謎の国家の法令"
        return "古典童話"

    categories = [_category(t) for t in titles]
    previews = [short_preview(t, 100) for t in texts]

    if view_dim == "2D":
        coords = project_to_2d(embeddings, method="pca")
        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "title": titles,
            "category": categories,
            "preview": previews,
        })
        fig_sc = px.scatter(
            df,
            x="x", y="y",
            color="category",
            text="title",
            hover_data={"preview": True, "x": False, "y": False},
            color_discrete_map={"古典童話": "#d62728", "謎の国家の法令": "#1f77b4"},
        )
        fig_sc.update_traces(
            marker={"size": 16, "line": {"width": 1, "color": "black"}},
            textposition="top center",
        )
        fig_sc.update_layout(
            height=540,
            xaxis_title="PC1",
            yaxis_title="PC2",
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
        )
        st.plotly_chart(fig_sc, width="stretch")
    else:
        coords3d = project_to_3d(embeddings, method="pca")
        df3 = pd.DataFrame({
            "x": coords3d[:, 0],
            "y": coords3d[:, 1],
            "z": coords3d[:, 2],
            "title": titles,
            "category": categories,
            "preview": previews,
        })
        fig_3d = px.scatter_3d(
            df3,
            x="x", y="y", z="z",
            color="category",
            text="title",
            hover_data={"preview": True, "x": False, "y": False, "z": False},
            color_discrete_map={"古典童話": "#d62728", "謎の国家の法令": "#1f77b4"},
        )
        fig_3d.update_traces(
            marker={"size": 9, "line": {"width": 1, "color": "black"}},
        )
        fig_3d.update_layout(
            height=620,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            scene={
                "xaxis_title": "PC1",
                "yaxis_title": "PC2",
                "zaxis_title": "PC3",
            },
        )
        st.plotly_chart(fig_3d, width="stretch")

    st.caption(
        "▶ 童話どうし・法令どうしがそれぞれ集まり、二つのクラスタに分かれます。"
        "RAG が法律質問に対して童話を引かないのは、まさにこの距離のおかげ。"
    )

    # --- (c) interactive single comparison ---
    st.markdown("**7-c. 任意の二つを選んで類似度を比べる**")
    col_a, col_b, col_metric = st.columns([1, 1, 1])
    with col_a:
        title_a = st.selectbox(
            "物語 A", options=titles, index=0, key="viz_story_a"
        )
    with col_b:
        title_b = st.selectbox(
            "物語 B", options=titles, index=1, key="viz_story_b"
        )
    i = titles.index(title_a)
    j = titles.index(title_b)
    s = float(sim[i, j])
    with col_metric:
        st.metric("コサイン類似度", f"{s:.4f}")
        st.write(_interpret_similarity(s))

    st.markdown("**どこが似ているか ― 実テキストのハイライト**")
    render_similarity_highlight(
        texts[i],
        texts[j],
        label_a=title_a,
        label_b=title_b,
        key_prefix="storylab",
    )

    # --- (d) cross-domain probe: pick a folktale, find closest corpus chunks ---
    st.markdown("**7-d. 童話 → 謎の国家の書庫 ハイブリッド検索**")
    st.caption(
        "童話をクエリにして書庫を検索すると何が起きるか。"
        "ジャンルが違いすぎてヒットは弱いはず、というのが意味的境界の証拠です。"
    )
    folktale_only_titles = [
        t for t in titles if not t.startswith("謎の国家")
    ]
    folktale_choice = st.selectbox(
        "童話を選ぶ", options=folktale_only_titles, index=0,
        key="viz_story_cross_probe",
    )
    probe_idx = titles.index(folktale_choice)
    probe_emb = embeddings[probe_idx]

    corpus_for_probe = _load_corpus()
    probe_sims = cosine_similarity_matrix(
        probe_emb[None, :], corpus_for_probe["embeddings"]
    )[0]
    probe_top = np.argsort(-probe_sims)[:5]

    rows = []
    for rank, idx_top in enumerate(probe_top, 1):
        meta = corpus_for_probe["metadatas"][idx_top]
        rows.append({
            "順位": rank,
            "類似度": float(probe_sims[idx_top]),
            "グループ": corpus_for_probe["labels"][idx_top],
            "Source": meta.get("source", ""),
            "Case ID": meta.get("case_id", ""),
            "プレビュー": corpus_for_probe["previews"][idx_top],
        })
    df_probe = pd.DataFrame(rows)
    st.dataframe(
        df_probe,
        width="stretch",
        hide_index=True,
        column_config={
            "類似度": st.column_config.ProgressColumn(
                "類似度", format="%.3f", min_value=0.0, max_value=1.0
            ),
        },
    )

    max_sim = float(probe_sims.max())
    avg_sim = float(probe_sims.mean())
    col_x, col_y = st.columns(2)
    col_x.metric("最大類似度（最も近い書庫文書）", f"{max_sim:.3f}")
    col_y.metric("平均類似度（書庫全体）", f"{avg_sim:.3f}")
    if max_sim < 0.45:
        st.success(
            f"✅ 最も近い文書でも類似度 {max_sim:.3f} と低めです。"
            "RAG が「童話」と「謎の国家の法令」をきちんと区別できている証拠。"
        )
    else:
        st.warning(
            f"⚠ 最も近い文書の類似度が {max_sim:.3f} と意外に高い。"
            "RAG が誤って引いてしまうリスクがある領域です。"
        )

    # --- (e) Network graph view ---
    st.markdown("**7-e. ネットワークグラフ — エッジは類似度の強さ**")
    st.caption(
        "閾値以上の類似度を持つ物語同士を線で結びます。"
        "童話のクラスタと法令のクラスタが切れた島になるはず。"
    )
    threshold = st.slider(
        "エッジを表示する類似度の閾値",
        min_value=0.20,
        max_value=0.90,
        value=0.50,
        step=0.05,
        key="viz_story_network_threshold",
    )

    # Layout the nodes using PCA projection
    coords_net = project_to_2d(embeddings, method="pca")

    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_weights: list[float] = []
    for ii in range(len(titles)):
        for jj in range(ii + 1, len(titles)):
            s_ij = float(sim[ii, jj])
            if s_ij < threshold:
                continue
            edge_x.extend([coords_net[ii, 0], coords_net[jj, 0], None])
            edge_y.extend([coords_net[ii, 1], coords_net[jj, 1], None])
            edge_weights.append(s_ij)

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 1, "color": "rgba(100, 100, 100, 0.4)"},
        hoverinfo="none",
        name="エッジ",
        showlegend=False,
    ))
    node_colors = [
        "#d62728" if not t.startswith("謎の国家") else "#1f77b4"
        for t in titles
    ]
    fig_net.add_trace(go.Scatter(
        x=coords_net[:, 0],
        y=coords_net[:, 1],
        mode="markers+text",
        marker={
            "size": 20,
            "color": node_colors,
            "line": {"width": 2, "color": "black"},
        },
        text=titles,
        textposition="top center",
        hovertext=[short_preview(t, 100) for t in texts],
        hoverinfo="text",
        name="ノード",
        showlegend=False,
    ))
    fig_net.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_net, width="stretch")

    n_edges = len(edge_weights)
    if n_edges == 0:
        st.info(
            f"閾値 {threshold:.2f} 以上のペアがありません。閾値を下げてください。"
        )
    else:
        avg_edge = sum(edge_weights) / n_edges
        st.caption(
            f"閾値 {threshold:.2f} 以上で結ばれたペア: {n_edges} 本、"
            f"平均類似度: {avg_edge:.3f}"
        )


# ---------------------------------------------------------------------------
# Section 8: Vector Arithmetic Lab
# ---------------------------------------------------------------------------


_ARITHMETIC_PRESETS: dict[str, dict[str, str]] = {
    "刑法 + 量刑 = ?": {
        "A": "謎の国家の刑法は、犯罪の構成要件、違法性、有責性を定める基本法である。",
        "B": "判決における量刑の判断基準として、被害の重大性、再犯の有無、改悛の情を考慮する。",
        "expect": "刑事判例（特に量刑が争点となるもの）に近づく",
    },
    "民法 + 損害 = ?": {
        "A": "謎の国家の民法は契約、不法行為、物権、相続など私人間の法律関係を規律する。",
        "B": "故意又は過失によって他人の権利を侵害したことから生じた損害賠償の責任。",
        "expect": "不法行為に関する民事判例に近づく",
    },
    "憲法 + 表現 = ?": {
        "A": "謎の国家憲法は基本的人権の尊重と統治機構を定める最高法規である。",
        "B": "言論、出版、集会、結社の自由は民主主義の根幹をなす。",
        "expect": "表現の自由・名誉毀損に関する憲法判例に近づく",
    },
    "桃太郎 + 海 = ?": {
        "A": "桃から生まれた男の子が動物を連れて鬼ヶ島へ鬼退治に行く昔話。",
        "B": "海・船・水中世界・潮流が物語の主舞台となる要素。",
        "expect": "浦島太郎（海＋主人公の冒険）に近づく",
    },
}


def _render_vector_arithmetic() -> None:
    st.subheader("8. ベクトル算術ラボ ― 文書ベクトルを足し引きすると？")
    st.caption(
        "「王 − 男 + 女 ≒ 女王」のような単語ベクトル算術の文章版。"
        "二つの文書ベクトルを線形結合（A + B、A − B、(A+B)/2）して、"
        "コーパス上で最も近い文書を探します。"
        "RAG が意味の足し算をどう処理するか見える化します。"
    )

    preset_name = st.selectbox(
        "プリセット",
        options=["(自由入力)"] + list(_ARITHMETIC_PRESETS.keys()),
        index=1,
        key="viz_arith_preset",
    )

    if preset_name == "(自由入力)":
        default_a = ""
        default_b = ""
        expect = ""
    else:
        preset = _ARITHMETIC_PRESETS[preset_name]
        default_a = preset["A"]
        default_b = preset["B"]
        expect = preset["expect"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**ベクトル A**")
        text_a = st.text_area(
            "A の本文",
            value=default_a,
            height=130,
            key=f"viz_arith_text_a_{preset_name}",
        )
    with col_b:
        st.markdown("**ベクトル B**")
        text_b = st.text_area(
            "B の本文",
            value=default_b,
            height=130,
            key=f"viz_arith_text_b_{preset_name}",
        )

    if not text_a.strip() or not text_b.strip():
        st.info("A と B の両方に本文を入力してください。")
        return

    operation = st.radio(
        "演算",
        options=["A + B (両方を兼ね備える)", "A - B (Aから B を引く)",
                 "(A + B) / 2 (中間点)"],
        index=0,
        horizontal=True,
        key="viz_arith_op",
    )

    if expect:
        st.caption(f"💡 期待される結果: **{expect}**")

    emb = _embed_cached((text_a.strip(), text_b.strip()))
    if operation.startswith("A + B"):
        combined = emb[0] + emb[1]
    elif operation.startswith("A - B"):
        combined = emb[0] - emb[1]
    else:
        combined = (emb[0] + emb[1]) / 2.0

    # Search corpus
    corpus = _load_corpus()
    sims = cosine_similarity_matrix(
        combined[None, :], corpus["embeddings"]
    )[0]
    top_idx = np.argsort(-sims)[:8]
    top_scores = sims[top_idx]

    st.markdown("**結果ベクトルに最も近いコーパス文書**")
    rows = []
    for rank, (i, score) in enumerate(zip(top_idx, top_scores), 1):
        meta = corpus["metadatas"][i]
        rows.append({
            "順位": rank,
            "類似度": float(score),
            "グループ": corpus["labels"][i],
            "Source": meta.get("source", ""),
            "Case ID": meta.get("case_id", ""),
            "プレビュー": corpus["previews"][i],
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "類似度": st.column_config.ProgressColumn(
                "類似度", format="%.3f", min_value=0.0, max_value=1.0
            ),
        },
    )

    # Side-check: how close is the result to A vs B alone?
    sim_to_a = cosine_similarity(combined, emb[0])
    sim_to_b = cosine_similarity(combined, emb[1])
    col1, col2, col3 = st.columns(3)
    col1.metric("結果 vs A", f"{sim_to_a:.3f}")
    col2.metric("結果 vs B", f"{sim_to_b:.3f}")
    col3.metric("A vs B", f"{cosine_similarity(emb[0], emb[1]):.3f}")


# ---------------------------------------------------------------------------
# Section 9: Retrieval Pipeline Lab  (dense vs hybrid vs rerank)
# ---------------------------------------------------------------------------


_PIPELINE_PRESETS: list[str] = [
    "刑法第235条が適用された判例は？",
    "窃盗で再犯だった場合の量刑",
    "正当防衛が認められた事例",
    "契約の取消しと無効の違い",
    "表現の自由が制限された憲法判例",
]


@st.cache_data(show_spinner="検索パイプラインを実行中…")
def _run_pipeline_cached(
    query: str,
    use_hybrid: bool,
    use_rerank: bool,
    k: int,
    fetch_k: int,
    dense_weight: float,
) -> dict[str, Any]:
    """Run the staged retrieval pipeline and return a plain, cacheable dict.

    Returning primitives (not LangChain Documents) keeps the Streamlit cache
    clean and the UI rendering trivial.
    """
    from rag_system.retriever import load_vectorstore, retrieve_advanced

    vs = load_vectorstore()
    staged = retrieve_advanced(
        query,
        vectorstore=vs,
        k=k,
        fetch_k=fetch_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        dense_weight=dense_weight,
    )

    def _label(doc: Any) -> str:
        m = doc.metadata or {}
        return m.get("case_id") or m.get("filename") or m.get("source") or "?"

    def _grp(doc: Any) -> str:
        return group_label(doc.metadata or {})

    dense_rows = [
        {
            "rank": i + 1,
            "label": _label(doc),
            "group": _grp(doc),
            "score": float(score),
            "preview": short_preview(doc.page_content, max_chars=90),
        }
        for i, (doc, score) in enumerate(staged.dense)
    ]
    sparse_rows = [
        {
            "rank": i + 1,
            "label": _label(doc),
            "group": _grp(doc),
            "score": float(score),
            "preview": short_preview(doc.page_content, max_chars=90),
        }
        for i, (doc, score) in enumerate(staged.sparse)
    ]
    fused_rows = [
        {
            "rank": i + 1,
            "label": _label(fr.document),
            "group": _grp(fr.document),
            "rrf": float(fr.rrf_score),
            "dense_rank": fr.dense_rank,
            "sparse_rank": fr.sparse_rank,
            "preview": short_preview(fr.document.page_content, max_chars=90),
        }
        for i, fr in enumerate(staged.fused)
    ]
    reranked_rows = [
        {
            "rank": rd.new_rank,
            "label": _label(rd.document),
            "group": _grp(rd.document),
            "score": float(rd.score),
            "original_rank": rd.original_rank,
            "delta": rd.rank_delta,
            "preview": short_preview(rd.document.page_content, max_chars=90),
        }
        for rd in staged.reranked
    ]

    return {
        "dense": dense_rows,
        "sparse": sparse_rows,
        "fused": fused_rows,
        "reranked": reranked_rows,
        "used_hybrid": staged.used_hybrid,
        "used_rerank": staged.used_rerank,
    }


def _render_pipeline_lab() -> None:
    st.subheader("9. 検索パイプライン比較ラボ ― 密 vs ハイブリッド vs リランク")
    st.caption(
        "実運用の RAG は「ベクトル検索でtop-kを取って終わり」ではありません。"
        "①密ベクトル検索で広めに候補を集め、②キーワード検索(BM25)と融合し、"
        "③cross-encoder で精密に並べ替える――この 3 段で精度を上げます。"
        "ここでは同じ質問に対して各段の結果がどう変わるかを並べて観察できます。"
    )

    with st.expander("🔍 なぜ密検索だけでは足りない？ 3 つの手法の違い", expanded=False):
        st.markdown(
            """
            | 手法 | 仕組み | 得意 | 苦手 |
            |------|--------|------|------|
            | **密ベクトル検索** | 質問と文書を別々にベクトル化し近さを測る（bi-encoder） | 言い換え・意味的な類似 | 「刑法第235条」のような**正確なキーワード**を取りこぼす |
            | **BM25（疎・キーワード）** | 単語の一致頻度でスコア | 条文番号・固有名詞の**完全一致** | 言い換え（「盗み」≠「窃盗」）に弱い |
            | **ハイブリッド（RRF 融合）** | 上記2つの**順位**を融合 | 両方の良いとこ取り | ― |
            | **リランク（cross-encoder）** | 質問と文書を**一緒に**Transformerへ入れ精密採点 | 細かな関連性判定 | 計算コスト（だから候補を絞ってから使う） |

            **ポイント:** 密検索は「だいたい近い」を高速に集めるのが役割。
            最後に cross-encoder で「本当に質問に答えているか」を採点し直すと、
            下位に埋もれていた正解が上位に来ます（順位の矢印で確認できます）。
            """
        )

    query = st.text_input(
        "質問を入力（条文番号を含む質問だとハイブリッドの効果が見えやすい）",
        value=_PIPELINE_PRESETS[0],
        key="viz_pipeline_query",
    )
    cols = st.columns(len(_PIPELINE_PRESETS))
    for col, preset in zip(cols, _PIPELINE_PRESETS):
        if col.button(preset, key=f"viz_pipe_preset_{preset}", width="stretch"):
            st.session_state["viz_pipeline_query"] = preset
            query = preset

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        k = st.slider("最終件数 k", 3, 10, 5, key="viz_pipe_k")
    with c2:
        fetch_k = st.slider("候補プール fetch_k", 10, 40, 20, step=5,
                            key="viz_pipe_fetchk")
    with c3:
        dense_weight = st.slider(
            "融合の重み（左=BM25寄り / 右=ベクトル寄り）",
            0.0, 1.0, 0.5, step=0.1, key="viz_pipe_w",
        )

    run = st.button(
        "🔎 検索パイプラインを実行", key="viz_pipe_run", type="primary"
    )
    st.caption(
        "※ 初回はリランカーモデルの読み込みで数秒かかります"
        "（cross-encoder を遅延ロードするため）。"
    )

    if not query.strip():
        st.info("質問を入力してください。")
        return
    if not run:
        st.info("上のボタンを押すと、3 つのパイプラインの結果を比較表示します。")
        return

    result = _run_pipeline_cached(
        query.strip(), True, True, k, fetch_k, dense_weight
    )

    if not result["used_rerank"]:
        st.warning(
            "リランカーモデルが読み込めなかったため、リランク段はスキップされています"
            "（密・ハイブリッドの比較は有効）。"
        )

    # --- Side-by-side: the three pipelines' final top-k ---
    st.markdown("#### 最終 top-k の比較（同じ質問、3 つのパイプライン）")

    dense_only = {r["label"]: r["rank"] for r in result["dense"][:k]}
    fused_only = {r["label"]: r["rank"] for r in result["fused"][:k]}
    rerank_final = {r["label"]: r["rank"] for r in result["reranked"][:k]}

    col_d, col_h, col_r = st.columns(3)
    with col_d:
        st.markdown("**① 密ベクトルのみ**")
        for r in result["dense"][:k]:
            st.markdown(
                f"`#{r['rank']}` **{r['label']}** ({r['group']})  \n"
                f"<span style='color:gray;font-size:0.8em'>近さ={r['score']:.3f}（高=近い） — {r['preview']}</span>",
                unsafe_allow_html=True,
            )
    with col_h:
        st.markdown("**② ＋ハイブリッド (BM25融合)**")
        if result["used_hybrid"]:
            for r in result["fused"][:k]:
                tag = _movement_tag(r["label"], dense_only, r["rank"])
                drk = r["dense_rank"] or "—"
                srk = r["sparse_rank"] or "—"
                st.markdown(
                    f"`#{r['rank']}` **{r['label']}** ({r['group']}) {tag}  \n"
                    f"<span style='color:gray;font-size:0.8em'>密#{drk}/疎#{srk} — {r['preview']}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("（ハイブリッド無効）")
    with col_r:
        st.markdown("**③ ＋リランク (cross-encoder)**")
        if result["used_rerank"]:
            for r in result["reranked"][:k]:
                tag = _movement_tag(r["label"], fused_only, r["rank"])
                st.markdown(
                    f"`#{r['rank']}` **{r['label']}** ({r['group']}) {tag}  \n"
                    f"<span style='color:gray;font-size:0.8em'>score={r['score']:.2f} — {r['preview']}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("（リランク無効）")

    st.caption(
        "🟢 = 前段より順位が上がった / 🔴 = 下がった / ✨ = 前段の top-k 圏外から浮上。"
        "リランクで ✨ が出たら「埋もれていた正解を救い上げた」好例です。"
    )

    # --- Rerank movement chart (the headline visual) ---
    if result["used_rerank"] and result["reranked"]:
        st.markdown("#### リランクによる順位変動（候補プール内）")
        rr = result["reranked"]
        labels = [f"#{r['original_rank']}→#{r['rank']} {r['label']}" for r in rr]
        deltas = [r["delta"] for r in rr]
        colors = ["#2ca02c" if d > 0 else ("#d62728" if d < 0 else "#888")
                  for d in deltas]
        fig = go.Figure(
            go.Bar(
                x=deltas,
                y=labels,
                orientation="h",
                marker_color=colors,
                hovertext=[r["preview"] for r in rr],
            )
        )
        fig.update_layout(
            height=max(300, 26 * len(rr)),
            xaxis_title="順位の上昇幅（右=順位アップ）",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "cross-encoder が「質問とどれだけ噛み合うか」で採点し直した結果。"
            "棒が右に長い文書ほど、密検索では低く見られていたが実は関連が強かったもの。"
        )


def _movement_tag(label: str, prev_ranks: dict[str, int], new_rank: int) -> str:
    """Return an emoji tag describing how ``label`` moved vs a previous stage."""
    prev = prev_ranks.get(label)
    if prev is None:
        return "✨"  # newly surfaced into the top-k
    if new_rank < prev:
        return f"🟢▲{prev - new_rank}"
    if new_rank > prev:
        return f"🔴▼{new_rank - prev}"
    return "="


# ---------------------------------------------------------------------------
# Page Entry Point
# ---------------------------------------------------------------------------


def render_visualization_page() -> None:
    """Render the vector visualization page (entry point used by app.py)."""
    st.header("ベクトル可視化")
    st.caption(
        "RAG は文章を「数字の並び（ベクトル）」として保持し、"
        "近い／遠いを測って関連文書を引いてきます。"
        "このページでは「謎の国家」の書庫がベクトル空間でどう見えているかを"
        "覗き見しながら、桃太郎と浦島太郎のような直感的な例も並べて遊べます。"
    )

    with st.expander("📖 はじめに ― 桃太郎と浦島太郎で理解する RAG", expanded=False):
        st.markdown(
            """
            **RAG（Retrieval-Augmented Generation）の心臓部はベクトルです。**

            文章を 768 個の数字に変換し、近い文章同士はベクトル空間で近く、
            違う文章同士は遠く配置されます。たとえば:

            | 比較 | コサイン類似度 | 意味 |
            |------|---------------|------|
            | 桃太郎 vs 浦島太郎 | **0.70** | やや似ている（旅・主人公・異界という共通モチーフ） |
            | 桃太郎 vs 三匹の子豚 | **0.58** | 「童話」というジャンルだけ共通、内容は別 |
            | 桃太郎 vs シンデレラ | **0.47** | さらに離れる |
            | 桃太郎 vs 謎の国家の刑法 | **0.11** | ジャンルが全く違う |
            | 謎の国家・刑法 vs 憲法 | **0.65** | 同じ法令同士で関連 |

            このページの各セクションは、上のような **「近い／遠い」** という
            感覚を、謎の国家の 1,354 チャンク（法令 + 判例）に対して
            様々な角度から可視化したものです。

            **おすすめの探索順序:**
            1. **書庫マップ** で全体を俯瞰（憲法・刑法・判例が別クラスタになるはず）
            2. 自分のクエリを **クエリ近傍探索** に投げてみる
            3. **物語ベクトルラボ** で童話の類似度を観察し、感覚を掴む
            4. **ベクトル算術ラボ** で「刑法 + 量刑 = ?」を試す

            より詳しい解説は `docs/VECTOR_SPACE.md` を参照してください。
            """
        )

    with st.expander("🗺️ 目次（10 セクション）", expanded=False):
        st.markdown(
            """
            0. **RAG フロー可視化** — 質問が回答になるまでの 4 工程（埋め込み→検索→プロンプト構築→生成）
            1. **書庫マップ** — 1,354 チャンクの 2D/3D 散布図
            2. **クエリ近傍探索** — 自由入力クエリの近傍 K 件と類似度分布
            3. **クラスタ自動発見** — K-means による自動グループ化と人手ラベル交差
            4. **グループ間類似度ヒートマップ** — 法令／判例グループ重心の距離マップ
            5. **文書近傍エクスプローラ** — コーパス内文書を起点に近傍検索
            6. **二文書比較ラボ** — 任意二文書のコサイン類似度と寄与次元
            7. **物語ベクトルラボ** — 桃太郎・浦島太郎・三匹の子豚デモ＋クロスドメインプローブ
            8. **ベクトル算術ラボ** — A + B、A − B、(A+B)/2 で意味の合成を試す
            9. **検索パイプライン比較ラボ** — 密 vs ハイブリッド(BM25融合) vs リランク(cross-encoder)を並べて比較

            **「RAG の基本を理解したい」なら最初は 0 から、**
            **「ベクトル幾何を見たい」なら 1 〜 8 を順に。**
            """
        )

    # Top-level stats
    try:
        stats = _stats()
    except Exception:
        st.error(
            "ChromaDB を読み込めません。"
            "先に `python -m rag_system.ingest` を実行してください。"
        )
        logger.exception("vector visualization page: collection load failed")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("総チャンク数", f"{stats['total']:,}")
    col2.metric("法令", f"{stats['by_document_type'].get('legal_framework', 0):,}")
    col3.metric("判例", f"{stats['by_document_type'].get('precedent', 0):,}")
    col4.metric("ベクトル次元", stats["dimension"])

    st.divider()

    # Lazy-load corpus only when one of the sections that needs it runs
    try:
        corpus = _load_corpus()
    except Exception:
        st.error(
            "コーパスのロードに失敗しました。"
            "ChromaDB の初期化を確認してください。"
        )
        logger.exception("vector visualization page: corpus load failed")
        return

    _render_rag_flow(corpus)
    st.divider()
    _render_archive_map(corpus)
    st.divider()
    _render_query_neighborhood(corpus)
    st.divider()
    _render_cluster_analysis(corpus)
    st.divider()
    _render_group_heatmap(corpus)
    st.divider()
    _render_neighbor_explorer(corpus)
    st.divider()
    _render_two_doc_comparison()
    st.divider()
    _render_story_lab()
    st.divider()
    _render_vector_arithmetic()
    st.divider()
    _render_pipeline_lab()

    st.divider()
    st.caption(
        "📘 ベクトル空間の概念や次元削減の読み解き方は "
        "`docs/VECTOR_SPACE.md` を、CLI のサンプル出力例は "
        "`docs/SAMPLE_OUTPUT.md` を参照してください。"
        "プログラムからの利用例は `scripts/vector_space_demo.py` にあります。"
    )
