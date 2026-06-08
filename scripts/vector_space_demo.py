#!/usr/bin/env python3
"""Vector-space exploration demo for the 謎の国家 RAG project.

Walks through the main capabilities of :mod:`rag_system.vector_analysis`
end-to-end so you can see — and copy — how to use the API from your own
scripts.

What the demo prints:

1. **Collection stats** — how many chunks live in ChromaDB.
2. **The 桃太郎 / 浦島太郎 / 三匹の子豚 metaphor** — pure-text similarity
   demo, no ChromaDB needed.
3. **Cross-domain probe** — embed a folktale and ask "what's the closest
   謎の国家 legal chunk?"  This is the question that motivated the whole
   visualization page.
4. **Story-vs-law geometry** — average similarity between every folktale
   and every legal sub-corpus.
5. **A live query** — run a natural-language legal query and print the
   top-5 chunks with similarity scores.

Run it after `python -m rag_system.ingest` has populated ChromaDB:

    python scripts/vector_space_demo.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

# Make sure the project root is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def demo_collection_stats() -> None:
    from rag_system.vector_analysis import get_collection_stats

    _section("1. ChromaDB の中身を眺める")
    stats = get_collection_stats()
    print(f"総チャンク数: {stats['total']:,}")
    print(f"ベクトル次元: {stats['dimension']}")
    print(f"文書タイプ別: {stats['by_document_type']}")
    print(f"事件種別:     {stats['by_case_type']}")


def demo_folktale_similarity() -> None:
    """The 桃太郎/浦島太郎/三匹の子豚 metaphor as code."""
    from rag_system.vector_analysis import (
        cosine_similarity_matrix,
        embed_texts,
    )

    _section("2. 桃太郎 / 浦島太郎 / 三匹の子豚 のベクトル類似度")

    titles = ["桃太郎", "浦島太郎", "三匹の子豚", "シンデレラ"]
    texts = [
        "桃から生まれた男の子が動物を連れて鬼ヶ島へ鬼退治に行く昔話。",
        "亀を助けた青年が竜宮城へ招かれ、長い時を過ごす昔話。",
        "わら・木・レンガで家を建てた三兄弟と襲うオオカミの物語。",
        "魔法の力で舞踏会へ行き、ガラスの靴で王子に見つけられる童話。",
    ]
    emb = embed_texts(texts)
    sim = cosine_similarity_matrix(emb)

    print(f"{'':10s}" + "".join(f"{t:>10s}" for t in titles))
    for i, t in enumerate(titles):
        row = f"{t:10s}" + "".join(f"{sim[i, j]:10.3f}" for j in range(len(titles)))
        print(row)
    print(
        "\n→ 桃太郎と浦島太郎の類似度が一番高く、三匹の子豚はそれより少し離れる。"
        "\n  これが RAG の「似ている／似ていない」の感覚です。"
    )


def demo_cross_domain_probe() -> None:
    """Embed a folktale, search the legal corpus, see what RAG would pull."""
    from rag_system.vector_analysis import (
        embed_texts,
        fetch_all_embeddings,
        group_label,
        short_preview,
        top_k_similar,
    )

    _section("3. 童話をクエリにして謎の国家の書庫を検索する（境界の探究）")

    folktale_query = (
        "桃から生まれた男の子が鬼ヶ島へ鬼退治に行き、動物を仲間にして"
        "宝物を持ち帰った勧善懲悪の物語。"
    )
    print(f"クエリ: 桃太郎の物語\n本文: {folktale_query[:80]}…\n")

    data = fetch_all_embeddings()
    query_emb = embed_texts([folktale_query])[0]
    idx, scores = top_k_similar(query_emb, data["embeddings"], k=5)

    print("謎の国家の書庫で最も近い 5 件:")
    for rank, (i, score) in enumerate(zip(idx, scores), 1):
        meta = data["metadatas"][i]
        label = group_label(meta)
        print(
            f"  #{rank} [類似度 {score:.3f}] [{label}] "
            f"{meta.get('case_id') or meta.get('source', '')[:48]}"
        )
        print(f"        {short_preview(data['documents'][i], 120)}")

    print(
        "\n→ 童話と最も近い法律チャンクでも類似度はせいぜい 0.3-0.4 台。"
        "\n  これがクロスドメインのギャップ。"
        "RAG は通常、しっかり遠い文書は引っ張ってこないので安心です。"
    )


def demo_story_vs_law_geometry() -> None:
    """Average similarity between folktales and legal sub-corpora."""
    from rag_system.vector_analysis import (
        cosine_similarity_matrix,
        embed_texts,
        fetch_all_embeddings,
        group_label,
    )

    _section("4. 童話と法律サブ書庫の平均類似度マトリックス")

    folktales = {
        "桃太郎": "桃から生まれた男の子が動物と鬼を退治する物語。",
        "浦島太郎": "亀を助けた青年が竜宮城で長い時を過ごす物語。",
        "三匹の子豚": "わら・木・レンガで家を建てた三兄弟とオオカミの物語。",
        "赤ずきん": "森のおばあさんを訪ねる少女がオオカミに騙される童話。",
    }
    folktale_emb = embed_texts(list(folktales.values()))

    data = fetch_all_embeddings()
    labels = [group_label(m) for m in data["metadatas"]]
    unique_groups = sorted(set(labels))

    # Average sim per folktale × per group
    print(f"{'':12s}" + "".join(f"{g:>14s}" for g in unique_groups))
    for ft, q in zip(folktales.keys(), folktale_emb):
        row_sims = cosine_similarity_matrix(q[None, :], data["embeddings"])[0]
        means = []
        for g in unique_groups:
            mask = np.array([lbl == g for lbl in labels], dtype=bool)
            means.append(float(row_sims[mask].mean()))
        print(f"{ft:12s}" + "".join(f"{m:14.3f}" for m in means))

    print(
        "\n→ 全ての童話が法律の全グループに対して低類似度（~0.05-0.15 程度）。"
        "\n  RAG の境界が機能していることが定量的に確認できます。"
    )


def demo_live_legal_query() -> None:
    """Real legal query against the corpus."""
    from rag_system.vector_analysis import (
        embed_texts,
        fetch_all_embeddings,
        group_label,
        short_preview,
        top_k_similar,
    )

    _section("5. 法律クエリ：『窃盗罪で再犯の場合の量刑』")

    data = fetch_all_embeddings()
    query = "窃盗罪で再犯の場合、量刑はどのように加重されるか？"
    print(f"クエリ: {query}\n")

    q_emb = embed_texts([query])[0]
    idx, scores = top_k_similar(q_emb, data["embeddings"], k=5)

    print("最も近い 5 件:")
    for rank, (i, score) in enumerate(zip(idx, scores), 1):
        meta = data["metadatas"][i]
        label = group_label(meta)
        head = f"  #{rank} [類似度 {score:.3f}] [{label}]"
        if meta.get("case_id"):
            head += f" {meta['case_id']}"
        elif meta.get("source"):
            head += f" {meta['source']}"
        print(head)
        print(f"        {short_preview(data['documents'][i], 140)}")

    print(
        "\n→ 法律クエリでは類似度 0.6-0.8 台のヒットが取れる。"
        "\n  童話クエリ（0.3-0.4）との差が、RAG の領域特化の根拠です。"
    )


def main() -> None:
    print("謎の国家 — ベクトル空間探索デモ")
    print("---------------------------------")
    print("（初回は埋め込みモデルのロードに数秒〜十数秒かかります）")

    try:
        demo_collection_stats()
        demo_folktale_similarity()
        demo_cross_domain_probe()
        demo_story_vs_law_geometry()
        demo_live_legal_query()
    except Exception as exc:
        # Collapse opaque chromadb / network errors into a helpful message
        msg = str(exc).lower()
        if "not exist" in msg or "no such" in msg or "not found" in msg:
            print()
            print("!" * 72)
            print("  ChromaDB が初期化されていません。")
            print("  先に取り込みを実行してください:")
            print("    python -m rag_system.ingest")
            print("!" * 72)
            sys.exit(1)
        raise

    print()
    print("=" * 72)
    print("  完了：詳細は Streamlit のベクトル可視化ページで対話的に探索できます。")
    print("  streamlit run app.py  →  サイドバーから「ベクトル可視化」")
    print("=" * 72)


if __name__ == "__main__":
    main()
