"""類似度＆解析ページ ― 6タブで埋め込み空間を探索する。

埋め込み空間（768次元 × 全文書）を多角的に探索する解析画面の本体。
``render_analysis_page()`` がエントリで、app.py の「類似度＆解析」ページと、
highlight_tool.py の単体起動の両方から共通で呼ばれる。

  🎨 ハイライト   … 2テキストの「どこが似ているか」を3方式で色付け
  🧭 意味軸       … PCA主成分・概念軸とその分散
  🗺️ 書庫マップ   … 全文書を2D/3D散布図で配置（PCA/t-SNE, グループ色分け）
  🔍 クエリ探索   … 任意の質問で近傍検索＋クエリと文書の語ハイライト
  🎯 クラスタ発見 … K-meansで意味グループを自動発見、人手ラベルと比較
  🔗 重複検出     … 似すぎる文書ペアを抽出（冗長性チェック）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ui.highlight_view import render_similarity_highlight

GROUP_COLORS: dict[str, str] = {
    "憲法": "#1f77b4", "刑法": "#d62728", "民法": "#2ca02c",
    "行政法": "#9467bd", "文化規制": "#ff7f0e", "倫理指針": "#17becf",
    "判例 (刑事)": "#e377c2", "判例 (民事)": "#8c564b", "判例 (憲法)": "#7f7f7f",
    "日本昔話": "#ff9896", "世界童話": "#aec7e8", "その他": "#bcbd22",
}

_PRESETS: dict[str, str] = {
    "(自由入力)": "",
    "桃太郎": (
        "むかしむかし、おばあさんが川で洗濯をしていると大きな桃が流れてきた。"
        "桃を割ると元気な男の子が現れた。男の子は桃太郎と名付けられ、やがて鬼ヶ島へ"
        "鬼退治の旅に出た。イヌ・サル・キジを家来にして鬼を倒し、宝物を持ち帰った。"
    ),
    "浦島太郎": (
        "むかしむかし、浦島太郎という若い漁師がいた。ある日、子供たちにいじめられて"
        "いた亀を助けた。亀のお礼に竜宮城へ連れて行かれ、乙姫様と楽しく過ごした。"
        "玉手箱を開けると白い煙が出て老人になってしまった。"
    ),
    "三匹の子豚": (
        "三匹の子豚兄弟が独立して家を建てた。一番目はわらの家、二番目は木の家、"
        "三番目はレンガの家。オオカミがやってきて、わらの家と木の家を吹き飛ばしたが、"
        "レンガの家だけは壊せなかった。"
    ),
    "シンデレラ": (
        "継母と義姉にいじめられていたシンデレラ。魔法使いに助けられ、舞踏会へ行き"
        "王子と出会う。12時の鐘で慌てて帰る途中、ガラスの靴を片方落としてしまう。"
        "王子はその靴を頼りにシンデレラを探し出した。"
    ),
    "刑法235条(窃盗)": (
        "謎の国家の刑法第235条によれば、他人の財物を窃取した者は窃盗罪に問われる。"
        "量刑は10年以下の懲役または50万円以下の罰金とする。"
    ),
    "民法709条(不法行為)": (
        "謎の国家の民法第709条によれば、故意又は過失によって他人の権利又は法律上"
        "保護される利益を侵害した者は、これによって生じた損害を賠償する責任を負う。"
    ),
}


def _label(meta: dict) -> str:
    """物語は genre（日本昔話/世界童話）、それ以外は通常の group_label。"""
    from rag_system.vector_analysis import group_label
    if meta.get("document_type") == "story":
        return meta.get("genre", "物語")
    return group_label(meta)


@st.cache_data(show_spinner="書庫(ChromaDB)を読み込み中…")
def _load_corpus() -> dict[str, object]:
    from rag_system.vector_analysis import fetch_all_embeddings, short_preview
    data = fetch_all_embeddings()
    labels = [_label(m) for m in data["metadatas"]]
    previews = [short_preview(d, 120) for d in data["documents"]]
    return {
        "embeddings": data["embeddings"], "labels": labels,
        "documents": data["documents"], "metadatas": data["metadatas"],
        "previews": previews,
    }


@st.cache_data(show_spinner="埋め込み計算中…")
def _embed_query(text: str) -> np.ndarray:
    from rag_system.vector_analysis import embed_texts
    return embed_texts([text])[0]


@st.cache_data(show_spinner="次元削減中…")
def _project(emb_bytes: bytes, shape: tuple, n: int, method: str) -> np.ndarray:
    from rag_system.vector_analysis import project_to_2d, project_to_3d
    emb = np.frombuffer(emb_bytes, dtype=np.float32).reshape(shape)
    return project_to_2d(emb, method) if n == 2 else project_to_3d(emb, method)


@st.cache_data(show_spinner="クラスタリング中…")
def _kmeans(emb_bytes: bytes, shape: tuple, k: int):
    from rag_system.vector_analysis import kmeans_clusters
    emb = np.frombuffer(emb_bytes, dtype=np.float32).reshape(shape)
    return kmeans_clusters(emb, n_clusters=k)


def _tab_highlight() -> None:
    st.caption(
        "2つのテキストの『どこが似ているか』を 📑文アライメント／🔥ヒートマップ／"
        "🔗ペアリングの3方式で色付け。コサイン類似度という1つの数字を分解します。")
    c1, c2 = st.columns(2)
    with c1:
        pa = st.selectbox("プリセット A", list(_PRESETS), index=1, key="preset_a")
        ta = st.text_area("テキスト A", _PRESETS[pa], height=200, key=f"ta_{pa}")
    with c2:
        pb = st.selectbox("プリセット B", list(_PRESETS), index=2, key="preset_b")
        tb = st.text_area("テキスト B", _PRESETS[pb], height=200, key=f"tb_{pb}")
    la = pa if pa != "(自由入力)" else "テキスト A"
    lb = pb if pb != "(自由入力)" else "テキスト B"
    st.divider()
    if ta.strip() and tb.strip():
        render_similarity_highlight(ta, tb, label_a=la, label_b=lb, key_prefix="tool")
    else:
        st.info("両方のテキストを入力すると、似ている箇所が色付けされます。")


def _tab_axes(corpus: dict) -> None:
    from rag_system.semantic_axes import (
        axis_variance_by_group, concept_axis, explained_variance, project_on_axis)
    E, labels = corpus["embeddings"], corpus["labels"]
    st.caption("全文書埋め込みから「意味のある軸」を抽出し、軸上の分布（分散）を見ます。")
    ev = explained_variance(E, n_components=20)
    st.markdown("#### 📊 PCA 説明分散比")
    st.bar_chart(pd.DataFrame({"説明率(%)": ev["ratio"] * 100},
                 index=[f"第{i+1}軸" for i in range(len(ev["ratio"]))]), height=220)
    if ev["cumulative"].size >= 20:
        st.caption(f"累積: 上位5軸={ev['cumulative'][4]*100:.0f}% / "
                   f"10軸={ev['cumulative'][9]*100:.0f}% / 20軸={ev['cumulative'][19]*100:.0f}%")
    st.divider()
    st.markdown("#### 🧭 概念軸（2グループの重心差）")
    groups = sorted(set(labels))
    cp, cn = st.columns(2)
    pos = cp.multiselect("正側(+)", groups, default=[g for g in groups if "判例" not in g], key="ax_pos")
    neg = cn.multiselect("負側(−)", groups, default=[g for g in groups if "判例" in g], key="ax_neg")
    if pos and neg:
        ax = concept_axis(E, np.array([l in pos for l in labels]),
                          np.array([l in neg for l in labels]))
        stats = axis_variance_by_group(project_on_axis(E, ax), labels)
        rows = [{"グループ": g, "位置(平均)": round(s["mean"], 3),
                 "広がり(±std)": round(s["std"], 3), "件数": s["count"]}
                for g, s in sorted(stats.items(), key=lambda kv: -kv[1]["mean"])]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("正側・負側のグループを選んでください。")


def _tab_map(corpus: dict) -> None:
    E = corpus["embeddings"]
    labels, previews = corpus["labels"], corpus["previews"]
    st.caption("全文書を2D/3Dに圧縮して配置。近い点は似た意味。グループで色分けします。"
               "物語（昔話/童話）は法律・判例から離れた独自の島になるはずです。")
    c1, c2, c3 = st.columns(3)
    dims = c1.radio("次元", ["2D", "3D"], horizontal=True, key="map_dims")
    method = c2.radio("削減手法", ["pca", "tsne"], horizontal=True, key="map_method",
                      format_func=str.upper, help="PCAは高速・全体構造、t-SNEは局所クラスタが鮮明")
    sel = c3.multiselect("表示グループ", sorted(set(labels)),
                         default=sorted(set(labels)), key="map_groups")
    mask = np.array([l in sel for l in labels], dtype=bool)
    if not mask.any():
        st.info("グループを選んでください。"); return
    emb = np.ascontiguousarray(E[mask], dtype=np.float32)
    n = 2 if dims == "2D" else 3
    coords = _project(emb.tobytes(), emb.shape, n, method)
    df = pd.DataFrame({"group": [l for l, k in zip(labels, mask) if k],
                       "preview": [p for p, k in zip(previews, mask) if k]})
    df["x"], df["y"] = coords[:, 0], coords[:, 1]
    if n == 2:
        fig = px.scatter(df, x="x", y="y", color="group", color_discrete_map=GROUP_COLORS,
                         hover_data={"preview": True, "x": False, "y": False}, opacity=0.7)
        fig.update_traces(marker={"size": 6})
    else:
        df["z"] = coords[:, 2]
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="group",
                            color_discrete_map=GROUP_COLORS,
                            hover_data={"preview": True, "x": False, "y": False, "z": False},
                            opacity=0.7)
        fig.update_traces(marker={"size": 3})
    fig.update_layout(height=600, margin={"l": 0, "r": 0, "t": 10, "b": 0}, legend_title="グループ")
    st.plotly_chart(fig, width="stretch")


def _tab_query(corpus: dict) -> None:
    from rag_system.vector_analysis import top_k_similar
    E = corpus["embeddings"]
    docs, labels, metas = corpus["documents"], corpus["labels"], corpus["metadatas"]
    st.caption("任意の質問を埋め込み、書庫の近い文書を順に表示。選んだ文書とクエリの"
               "『どの語が効いたか』を色付けします（検索の中身が見える）。")
    presets = ["窃盗罪の量刑は？", "正当防衛の成立要件", "鬼を退治する話",
               "亀を助ける物語", "魔法で舞踏会へ行く", "労働時間の上限"]
    q = st.selectbox("プリセット質問", ["(自由入力)"] + presets, index=1, key="q_preset")
    query = st.text_input("クエリ", q if q != "(自由入力)" else "", key=f"q_{q}")
    k = st.slider("取得件数", 3, 20, 8, key="q_k")
    if not query.strip():
        st.info("質問を入力してください。"); return
    qemb = _embed_query(query.strip())
    idx, scores = top_k_similar(qemb, E, k=k)
    rows = [{"順位": r, "類似度": float(s), "グループ": labels[i],
             "Source": metas[i].get("source", ""), "本文": corpus["previews"][i]}
            for r, (i, s) in enumerate(zip(idx, scores), 1)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"類似度": st.column_config.ProgressColumn(
                     "類似度", format="%.3f", min_value=0.0, max_value=1.0)})
    st.divider()
    st.markdown("#### 🎨 クエリ × 選んだ文書 の語ハイライト")
    pick = st.selectbox("ハイライトする文書（近傍から選ぶ）",
                        [f"#{r} {labels[i]} ({float(s):.3f})" for r, (i, s) in enumerate(zip(idx, scores), 1)],
                        key="q_pick")
    pi = int(pick.split()[0].lstrip("#")) - 1
    doc = docs[int(idx[pi])]
    render_similarity_highlight(query.strip(), doc[:600], label_a="クエリ",
                                label_b="文書", key_prefix="query")


def _tab_cluster(corpus: dict) -> None:
    E = corpus["embeddings"]
    labels, previews = corpus["labels"], corpus["previews"]
    st.caption("メタデータを見ずに、ベクトルの近さだけで K-means が自動でグループ分け。"
               "人手ラベルとのズレに、隠れたサブトピックが現れます。")
    c1, c2 = st.columns([1, 3])
    k = c1.slider("クラスタ数 K", 2, 20, 8, key="cl_k")
    method = c1.radio("可視化", ["pca", "tsne"], key="cl_method", format_func=str.upper)
    emb = np.ascontiguousarray(E, dtype=np.float32)
    cl_labels, _ = _kmeans(emb.tobytes(), emb.shape, k)
    coords = _project(emb.tobytes(), emb.shape, 2, method)
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1],
                       "cluster": [f"C{c}" for c in cl_labels],
                       "human": labels, "preview": previews})
    with c2:
        fig = px.scatter(df, x="x", y="y", color="cluster",
                         hover_data={"human": True, "preview": True, "x": False, "y": False},
                         opacity=0.7)
        fig.update_traces(marker={"size": 5})
        fig.update_layout(height=480, margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(fig, width="stretch")
    st.markdown("**クラスタ × 人手ラベル のクロス集計**")
    st.dataframe(pd.crosstab(df["human"], df["cluster"]), width="stretch")


def _tab_dup(corpus: dict) -> None:
    from rag_system.vector_analysis import cosine_similarity_matrix
    E = corpus["embeddings"]
    labels, metas = corpus["labels"], corpus["metadatas"]
    st.caption("似すぎる文書ペア（冗長な可能性）を抽出。書庫の品質チェックや"
               "RAG の検索が重複に埋もれていないかの確認に使えます。")
    thr = st.slider("類似度しきい値", 0.80, 0.99, 0.92, 0.01, key="dup_thr")
    maxn = st.slider("最大ペア表示数", 5, 100, 30, key="dup_n")
    sim = cosine_similarity_matrix(E)
    iu = np.triu_indices(sim.shape[0], k=1)
    pair_scores = sim[iu]
    keep = pair_scores >= thr
    ii, jj, ss = iu[0][keep], iu[1][keep], pair_scores[keep]
    order = np.argsort(-ss)[:maxn]
    if order.size == 0:
        st.success(f"しきい値 {thr:.2f} 以上の重複ペアはありません。"); return
    rows = []
    for o in order:
        a, b = int(ii[o]), int(jj[o])
        rows.append({"類似度": float(ss[o]),
                     "文書A": f"[{labels[a]}] {metas[a].get('source','')}",
                     "文書B": f"[{labels[b]}] {metas[b].get('source','')}",
                     "A本文": corpus["previews"][a], "B本文": corpus["previews"][b]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"類似度": st.column_config.NumberColumn(format="%.4f")})
    st.caption(f"しきい値 {thr:.2f} 以上のペア: {int(keep.sum())} 件（上位 {min(maxn, order.size)} 件を表示）")


def render_analysis_page() -> None:
    """類似度＆解析ページを描画する（6タブ）。"""
    st.title("🔬 類似度＆解析")
    st.caption("埋め込み空間（768次元 × 全文書）を 6 つの角度から探索します。"
               "Ollama 不要。物語（昔話・童話）も書庫に同梱済み。")
    tabs = st.tabs(["🎨 ハイライト", "🧭 意味軸", "🗺️ 書庫マップ",
                    "🔍 クエリ探索", "🎯 クラスタ発見", "🔗 重複検出"])
    with tabs[0]:
        _tab_highlight()
    try:
        corpus = _load_corpus()
    except Exception as exc:  # pragma: no cover
        for t in tabs[1:]:
            with t:
                st.warning(f"書庫(ChromaDB)を読み込めません: {exc}\n\n"
                           "`python -m rag_system.ingest` で取り込んでください。")
        return
    with tabs[1]:
        _tab_axes(corpus)
    with tabs[2]:
        _tab_map(corpus)
    with tabs[3]:
        _tab_query(corpus)
    with tabs[4]:
        _tab_cluster(corpus)
    with tabs[5]:
        _tab_dup(corpus)
