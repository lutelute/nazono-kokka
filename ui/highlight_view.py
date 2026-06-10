"""二文書の「似ている箇所」を色付け表示する Streamlit コンポーネント。

3 つの見方をタブで切り替える:
1. 文アライメント ― 対応する文ペアを同色で塗る
2. トークンヒートマップ ― 各単語の寄与を暖色/寒色の濃淡で（中心化コントラスト）
3. トークンペアリング ― A の各単語と B の最類似単語を同色＋番号で結ぶ

エントリは :func:`render_similarity_highlight`（⑥二文書比較ラボ・⑦物語ラボ共通）。
matplotlib などの追加依存は使わない（Streamlit ネイティブ機能のみ）。
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from rag_system.text_alignment import (
    SentenceAlignment,
    TokenContributions,
    TokenPairing,
    sentence_alignment,
    token_contributions,
    token_pairing,
)

_PAIR_PALETTE: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#5254a3",
]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


@st.cache_data(show_spinner=False)
def _cached_sentence_align(text_a: str, text_b: str) -> SentenceAlignment:
    return sentence_alignment(text_a, text_b)


@st.cache_data(show_spinner=False)
def _cached_contributions(text_a: str, text_b: str) -> TokenContributions:
    return token_contributions(text_a, text_b)


@st.cache_data(show_spinner=False)
def _cached_pairing(text_a: str, text_b: str) -> TokenPairing:
    return token_pairing(text_a, text_b)


def render_similarity_highlight(
    text_a: str,
    text_b: str,
    *,
    label_a: str = "文書 A",
    label_b: str = "文書 B",
    key_prefix: str = "hl",
) -> None:
    """二文書の類似箇所ハイライトを 3 方式タブで描画する。"""
    if not text_a.strip() or not text_b.strip():
        st.info("2 つのテキストを入力すると、似ている箇所が色付けされます。")
        return

    tab_sent, tab_heat, tab_pair = st.tabs([
        "📑 文アライメント", "🔥 トークンヒートマップ", "🔗 トークンペアリング",
    ])

    with tab_sent:
        st.caption(
            f"対応する文ペアを同じ色で塗ります。**{label_a}** の各文に最も似た "
            f"**{label_b}** の文を結びます。色が同じ＝意味的に対応、灰色＝対応が弱い文。"
        )
        _render_sentence_align(text_a, text_b, label_a, label_b, key_prefix)

    with tab_heat:
        st.caption(
            "各単語が**コサイン類似度にどれだけ効いているか**を背景色で表します。"
            "🟥 暖色＝相手文に近づける語／🟦 寒色＝引き離す語。中心化コントラストで"
            "助詞などの機能語は自動的に薄くなり、内容語が浮きます。"
        )
        _render_heatmap(text_a, text_b, label_a, label_b)

    with tab_pair:
        st.caption(
            f"**{label_a}** の各単語と、**{label_b}** で最も意味の近い単語を "
            "同じ色＋番号で結びます（BERTScore 的な max-sim 対応）。"
        )
        _render_pairing(text_a, text_b, label_a, label_b, key_prefix)


def _render_sentence_align(
    text_a: str, text_b: str, label_a: str, label_b: str, key_prefix: str
) -> None:
    sa = _cached_sentence_align(text_a, text_b)
    if not sa.sentences_a or not sa.sentences_b:
        st.info("文に分割できませんでした。")
        return

    threshold = st.slider(
        "対応とみなす類似度のしきい値",
        min_value=0.0, max_value=0.9, value=0.40, step=0.05,
        key=f"{key_prefix}_{label_a}_{label_b}_sent_th",
        help="この値以上で対応している文ペアだけ色を付けます。",
    )

    a_colors: list[str | None] = []
    used_b: set[int] = set()
    for i in range(len(sa.sentences_a)):
        j = int(sa.a_to_b[i])
        if sa.a_to_b_score[i] >= threshold:
            a_colors.append(_PAIR_PALETTE[j % len(_PAIR_PALETTE)])
            used_b.add(j)
        else:
            a_colors.append(None)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{html.escape(label_a)}**")
        out = []
        for i, sent in enumerate(sa.sentences_a):
            color = a_colors[i]
            j = int(sa.a_to_b[i])
            score = float(sa.a_to_b_score[i])
            out.append(_sentence_block(
                sent, color,
                badge=(f"→{label_b}文{j + 1} ({score:.2f})"
                       if color else f"対応弱 ({score:.2f})")))
        st.markdown("\n".join(out), unsafe_allow_html=True)
    with col_b:
        st.markdown(f"**{html.escape(label_b)}**")
        out = []
        for j, sent in enumerate(sa.sentences_b):
            color = _PAIR_PALETTE[j % len(_PAIR_PALETTE)] if j in used_b else None
            out.append(_sentence_block(sent, color, badge=f"文{j + 1}"))
        st.markdown("\n".join(out), unsafe_allow_html=True)

    with st.expander("文 × 文 類似度マトリックス"):
        df = pd.DataFrame(
            sa.similarity,
            index=[f"{label_a}文{i + 1}" for i in range(len(sa.sentences_a))],
            columns=[f"{label_b}文{j + 1}" for j in range(len(sa.sentences_b))])
        # matplotlib 依存の background_gradient は使わず、素の表＋小数2桁で表示
        st.dataframe(df.round(2), width="stretch")


def _sentence_block(sentence: str, color: str | None, badge: str) -> str:
    esc = html.escape(sentence)
    if color:
        bg = _rgba(color, 0.22)
        border = color
        badge_html = (f'<span style="font-size:0.72rem; color:{border}; '
                      f'font-weight:600">{html.escape(badge)}</span>')
    else:
        bg = "rgba(0,0,0,0.04)"
        border = "#cccccc"
        badge_html = (f'<span style="font-size:0.72rem; color:#999">'
                      f'{html.escape(badge)}</span>')
    return (f'<div style="background:{bg}; border-left:4px solid {border}; '
            f'padding:6px 10px; margin:4px 0; border-radius:4px; '
            f'line-height:1.6">{esc}<br>{badge_html}</div>')


def _render_heatmap(
    text_a: str, text_b: str, label_a: str, label_b: str
) -> None:
    tc = _cached_contributions(text_a, text_b)

    c1, c2, c3 = st.columns(3)
    c1.metric("コサイン類似度", f"{tc.cosine:.4f}")
    c2.metric(f"{label_a} 語数", f"{len(tc.side_a.words)}")
    c3.metric(f"{label_b} 語数", f"{len(tc.side_b.words)}")

    vmax = max(
        float(np.abs(tc.side_a.centered).max()) if tc.side_a.centered.size else 0.0,
        float(np.abs(tc.side_b.centered).max()) if tc.side_b.centered.size else 0.0,
        1e-9)

    st.markdown(f"**{html.escape(label_a)}**")
    st.markdown(_heatmap_html(tc.side_a.words, tc.side_a.centered,
                              tc.side_a.is_punct, vmax), unsafe_allow_html=True)
    st.markdown(f"**{html.escape(label_b)}**")
    st.markdown(_heatmap_html(tc.side_b.words, tc.side_b.centered,
                              tc.side_b.is_punct, vmax), unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:0.8rem; color:#666; margin-top:8px">凡例: '
        '<span style="background:rgba(229,57,53,0.65); padding:0 6px; border-radius:3px">強く似ている</span>&nbsp;'
        '<span style="background:rgba(229,57,53,0.25); padding:0 6px; border-radius:3px">やや似ている</span>&nbsp;'
        '<span style="background:rgba(30,110,195,0.25); padding:0 6px; border-radius:3px">やや離れる</span>&nbsp;'
        '<span style="background:rgba(30,110,195,0.65); padding:0 6px; border-radius:3px">強く離れる</span></div>',
        unsafe_allow_html=True)

    with st.expander("寄与の内訳（厳密分解：総和＝コサイン類似度）"):
        st.caption(
            "「生の寄与」は各語の厳密な寄与で総和がコサイン類似度に一致します"
            "（特殊トークン分だけ僅差）。色付けには機能語を抑えた「中心化コントラスト」を使用。")
        order = np.argsort(-tc.side_a.centered)
        rows = [{"単語": tc.side_a.words[i],
                 "中心化コントラスト": float(tc.side_a.centered[i]),
                 "生の寄与": float(tc.side_a.raw[i])} for i in order[:20]]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={
                         "中心化コントラスト": st.column_config.NumberColumn(format="%.4f"),
                         "生の寄与": st.column_config.NumberColumn(format="%.4f")})


def _heat_color(value: float, vmax: float) -> str:
    t = max(-1.0, min(1.0, value / vmax))
    if t >= 0:
        return _rgba("#e53935", 0.10 + 0.62 * t)
    return _rgba("#1e6ec3", 0.10 + 0.62 * abs(t))


def _heatmap_html(
    words: list[str], centered: np.ndarray, is_punct: np.ndarray, vmax: float
) -> str:
    spans: list[str] = []
    for w, c, p in zip(words, centered, is_punct):
        esc = html.escape(w)
        if p:
            spans.append(f'<span style="color:#c8c8c8">{esc}</span>')
            if "\n" in w or w in ("。", "！", "？"):
                spans.append("<br>")
            continue
        bg = _heat_color(float(c), vmax)
        spans.append(f'<span style="background-color:{bg}; border-radius:3px; '
                     f'padding:1px 1px">{esc}</span>')
    body = "".join(spans)
    return ('<div style="line-height:2.1; font-size:1.05rem; border:1px solid #eee; '
            f'border-radius:6px; padding:10px 12px; margin-bottom:10px">{body}</div>')


def _render_pairing(
    text_a: str, text_b: str, label_a: str, label_b: str, key_prefix: str
) -> None:
    tp = _cached_pairing(text_a, text_b)
    if not tp.words_a or not tp.words_b:
        st.info("単語に分割できませんでした。")
        return

    col_n, col_th = st.columns(2)
    with col_n:
        top_n = st.slider("結ぶペア数（上位）", min_value=3, max_value=20,
                          value=8, step=1, key=f"{key_prefix}_pair_n")
    with col_th:
        threshold = st.slider("最低類似度", min_value=0.0, max_value=0.9,
                              value=0.35, step=0.05, key=f"{key_prefix}_pair_th")

    cand = [i for i in np.argsort(-tp.a_to_b_score)
            if not tp.is_punct_a[i] and tp.a_to_b_score[i] >= threshold][:top_n]

    a_color: dict[int, int] = {}
    a_rank: dict[int, int] = {}
    b_color: dict[int, int] = {}
    b_rank: dict[int, int] = {}
    for rank, i in enumerate(cand):
        j = int(tp.a_to_b[i])
        a_color[i] = rank
        a_rank[i] = rank + 1
        if j not in b_color:
            b_color[j] = rank
            b_rank[j] = rank + 1

    st.markdown(f"**{html.escape(label_a)}**")
    st.markdown(_pairing_html(tp.words_a, a_color, a_rank), unsafe_allow_html=True)
    st.markdown(f"**{html.escape(label_b)}**")
    st.markdown(_pairing_html(tp.words_b, b_color, b_rank), unsafe_allow_html=True)

    if cand:
        rows = []
        for rank, i in enumerate(cand, 1):
            j = int(tp.a_to_b[i])
            rows.append({"#": rank, f"{label_a} の語": tp.words_a[i],
                         f"{label_b} の語": tp.words_b[j],
                         "類似度": float(tp.a_to_b_score[i])})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={"類似度": st.column_config.ProgressColumn(
                         "類似度", format="%.3f", min_value=0.0, max_value=1.0)})
    else:
        st.info("しきい値以上のペアがありません。最低類似度を下げてください。")


def _pairing_html(
    words: list[str], color_map: dict[int, int], rank_map: dict[int, int]
) -> str:
    spans: list[str] = []
    for i, w in enumerate(words):
        esc = html.escape(w)
        if i in color_map:
            color = _PAIR_PALETTE[color_map[i] % len(_PAIR_PALETTE)]
            bg = _rgba(color, 0.30)
            sup = f'<sup style="color:{color}; font-weight:700">{rank_map[i]}</sup>'
            spans.append(f'<span style="background-color:{bg}; border-bottom:2px solid '
                         f'{color}; border-radius:3px; padding:1px 1px">{esc}</span>{sup}')
        else:
            spans.append(f'<span style="color:#777">{esc}</span>')
        if "\n" in w or w in ("。", "！", "？"):
            spans.append("<br>")
    body = "".join(spans)
    return ('<div style="line-height:2.4; font-size:1.05rem; border:1px solid #eee; '
            f'border-radius:6px; padding:10px 12px; margin-bottom:10px">{body}</div>')
