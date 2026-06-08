"""Vector analysis utilities for the RAG judicial system.

Provides primitives for visualizing and exploring the embedding space:

- :func:`fetch_all_embeddings` — pull every chunk + its 768-dim vector
  out of ChromaDB
- :func:`get_embedding_model` — load (and cache) the sentence-transformers
  encoder used at ingestion time, so live queries land in the same space
- :func:`embed_texts` — encode arbitrary text into the shared vector space
- :func:`project_to_2d` / :func:`project_to_3d` — dimensionality reduction
  via PCA or t-SNE for interactive scatter plots
- :func:`cosine_similarity_matrix` — pairwise cosine similarity
- :func:`top_k_similar` — nearest-neighbor lookup against a corpus
- :func:`precompute_projections` — write PCA / t-SNE coordinates to disk
  so the Streamlit page can load instantly on next start

The intuition: every document (and every query) is a point in 768-dim
space.  Two stories like *桃太郎* and *浦島太郎* both involve a young man,
a journey, and a non-human realm — so their vectors are close.
*三匹の子豚* is also a folktale but lacks the journey motif, so it sits
further away.  These utilities expose that geometry to the UI.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from rag_system.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDINGS_MODEL_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding Model (cached)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_embedding_model(model_name: str = EMBEDDINGS_MODEL_NAME):
    """Load and cache the sentence-transformers encoder.

    The same model used at ingestion time is reused so that live queries
    map into the same 768-dim space as the stored documents.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to the project's configured
        embedding model.

    Returns
    -------
    SentenceTransformer
        The loaded encoder.
    """
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer

    logger.info("埋め込みモデルをロード: %s", model_name)
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of texts into the shared vector space.

    Parameters
    ----------
    texts:
        Raw input strings.

    Returns
    -------
    numpy.ndarray
        ``(N, 768)`` array of embeddings, one row per input text.
    """
    if not texts:
        return np.empty((0, 768), dtype=np.float32)
    model = get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# ChromaDB Access
# ---------------------------------------------------------------------------


def _get_collection():
    """Open the persisted ChromaDB collection used by the RAG pipeline."""
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(CHROMA_COLLECTION_NAME)


def fetch_all_embeddings(
    limit: int | None = None,
    document_type: str | None = None,
    case_type: str | None = None,
) -> dict[str, Any]:
    """Pull every embedded chunk out of ChromaDB with metadata.

    Parameters
    ----------
    limit:
        Maximum number of rows to return.  ``None`` means all (~1,354).
    document_type:
        Optional filter on ``metadata.document_type`` (``"legal_framework"``
        or ``"precedent"``).
    case_type:
        Optional filter on ``metadata.case_type`` (``"criminal"``,
        ``"civil"``, ``"constitutional"``).

    Returns
    -------
    dict
        ``{"ids": list[str], "embeddings": np.ndarray, "documents":
        list[str], "metadatas": list[dict]}``.  ``embeddings`` has shape
        ``(N, dim)``.
    """
    collection = _get_collection()

    where: dict[str, Any] | None = None
    conditions: list[dict[str, Any]] = []
    if document_type is not None:
        conditions.append({"document_type": {"$eq": document_type}})
    if case_type is not None:
        conditions.append({"case_type": {"$eq": case_type}})
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    kwargs: dict[str, Any] = {
        "include": ["embeddings", "documents", "metadatas"],
    }
    if where is not None:
        kwargs["where"] = where
    if limit is not None:
        kwargs["limit"] = limit

    raw = collection.get(**kwargs)

    raw_embeddings = raw.get("embeddings")
    if raw_embeddings is None or len(raw_embeddings) == 0:
        embeddings = np.empty((0, 768), dtype=np.float32)
    else:
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

    return {
        "ids": list(raw.get("ids") or []),
        "embeddings": embeddings,
        "documents": list(raw.get("documents") or []),
        "metadatas": list(raw.get("metadatas") or []),
    }


def get_collection_stats() -> dict[str, Any]:
    """Summarize the collection: total count + per-type breakdown."""
    collection = _get_collection()
    raw = collection.get(include=["metadatas"])
    metas = raw.get("metadatas") or []

    by_doc_type: dict[str, int] = {}
    by_case_type: dict[str, int] = {}
    for m in metas:
        dt = m.get("document_type", "unknown")
        by_doc_type[dt] = by_doc_type.get(dt, 0) + 1
        ct = m.get("case_type")
        if ct:
            by_case_type[ct] = by_case_type.get(ct, 0) + 1

    return {
        "total": collection.count(),
        "by_document_type": by_doc_type,
        "by_case_type": by_case_type,
        "dimension": 768,
    }


# ---------------------------------------------------------------------------
# Dimensionality Reduction
# ---------------------------------------------------------------------------


_VALID_METHODS = {"pca", "tsne"}


def project_to_2d(
    embeddings: np.ndarray,
    method: str = "pca",
    *,
    random_state: int = 42,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Reduce ``(N, dim)`` embeddings to ``(N, 2)`` for plotting.

    Parameters
    ----------
    embeddings:
        High-dim input vectors.
    method:
        ``"pca"`` (fast, linear, preserves global structure) or
        ``"tsne"`` (slower, non-linear, surfaces local clusters).
    random_state:
        Seed for reproducible t-SNE / PCA randomization.
    perplexity:
        t-SNE perplexity hyperparameter.  Ignored for PCA.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` projected coordinates.
    """
    return _project(embeddings, n_components=2, method=method,
                    random_state=random_state, perplexity=perplexity)


def project_to_3d(
    embeddings: np.ndarray,
    method: str = "pca",
    *,
    random_state: int = 42,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Reduce ``(N, dim)`` embeddings to ``(N, 3)`` for 3D plotting."""
    return _project(embeddings, n_components=3, method=method,
                    random_state=random_state, perplexity=perplexity)


def _project(
    embeddings: np.ndarray,
    *,
    n_components: int,
    method: str,
    random_state: int,
    perplexity: float,
) -> np.ndarray:
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method は {_VALID_METHODS} のいずれかである必要があります: {method}"
        )
    if embeddings.shape[0] == 0:
        return np.empty((0, n_components), dtype=np.float32)
    if embeddings.shape[0] < n_components + 1:
        # Not enough points; pad to required shape
        return np.zeros((embeddings.shape[0], n_components), dtype=np.float32)

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(embeddings).astype(np.float32)

    # t-SNE
    from sklearn.manifold import TSNE

    effective_perplexity = max(5.0, min(perplexity, embeddings.shape[0] - 1))
    reducer = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
    )
    return reducer.fit_transform(embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(
    a: np.ndarray, b: np.ndarray | None = None
) -> np.ndarray:
    """Compute the pairwise cosine similarity matrix.

    Parameters
    ----------
    a:
        ``(N, dim)`` array of row-vectors.
    b:
        Optional ``(M, dim)`` array.  If ``None``, ``b = a`` and an
        ``(N, N)`` similarity matrix is returned.

    Returns
    -------
    numpy.ndarray
        Cosine similarities in ``[-1, 1]``.
    """
    if a.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    if b is None:
        b = a
    if b.size == 0:
        return np.empty((a.shape[0], 0), dtype=np.float32)

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    a_unit = a / a_norm
    b_unit = b / b_norm
    return (a_unit @ b_unit.T).astype(np.float32)


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if u_norm == 0 or v_norm == 0:
        return 0.0
    return float(np.dot(u, v) / (u_norm * v_norm))


def kmeans_clusters(
    embeddings: np.ndarray,
    n_clusters: int = 8,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run K-means and return per-point labels + cluster centroids.

    Useful for "what natural groupings emerge from the embeddings, ignoring
    the human-assigned document/case types?" — clusters often surface
    sub-topics (e.g. 窃盗系の判例 vs 暴力系の判例) that the metadata doesn't
    capture.

    Parameters
    ----------
    embeddings:
        ``(N, dim)`` input vectors.
    n_clusters:
        Number of K-means clusters to fit.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    tuple of ndarray
        ``(labels, centroids)`` — ``labels`` has shape ``(N,)`` of int64,
        ``centroids`` has shape ``(n_clusters, dim)``.
    """
    if embeddings.shape[0] == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, embeddings.shape[1] if embeddings.ndim == 2 else 0),
                     dtype=np.float32),
        )
    n_clusters = max(1, min(n_clusters, embeddings.shape[0]))

    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(embeddings)
    return labels.astype(np.int64), kmeans.cluster_centers_.astype(np.float32)


def top_k_similar(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the top-K most similar vectors in ``corpus_emb``.

    Parameters
    ----------
    query_emb:
        Either a single ``(dim,)`` query vector or a batch ``(Q, dim)``.
    corpus_emb:
        ``(N, dim)`` corpus to search against.
    k:
        Number of nearest neighbors to return.

    Returns
    -------
    tuple of ndarray
        ``(indices, scores)`` — for a single query both are 1-D of length
        ``k``; for a batch they are 2-D of shape ``(Q, k)``.
    """
    if corpus_emb.shape[0] == 0:
        if query_emb.ndim == 1:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return (np.empty((query_emb.shape[0], 0), dtype=np.int64),
                np.empty((query_emb.shape[0], 0), dtype=np.float32))

    if query_emb.ndim == 1:
        sims = cosine_similarity_matrix(query_emb[None, :], corpus_emb)[0]
        k = min(k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]

    sims = cosine_similarity_matrix(query_emb, corpus_emb)
    k = min(k, sims.shape[1])
    idx_full = np.argsort(-sims, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    return idx_full, sims[rows, idx_full]


# ---------------------------------------------------------------------------
# Helpers for UI
# ---------------------------------------------------------------------------


def group_label(metadata: dict[str, Any]) -> str:
    """Return a human-readable group label for a chunk's metadata.

    Used as the color/legend value in scatter plots.  Combines
    ``document_type`` with ``case_type`` when available, e.g.
    ``"判例 (刑事)"`` or ``"法令"``.
    """
    doc_type = metadata.get("document_type", "")
    case_type = metadata.get("case_type", "")

    if doc_type == "legal_framework":
        filename = metadata.get("filename", "")
        return {
            "constitution": "憲法",
            "criminal_code": "刑法",
            "civil_code": "民法",
            "administrative_code": "行政法",
            "cultural_regulations": "文化規制",
            "ethical_guidelines": "倫理指針",
        }.get(filename, "法令")

    if doc_type == "precedent":
        return {
            "criminal": "判例 (刑事)",
            "civil": "判例 (民事)",
            "constitutional": "判例 (憲法)",
        }.get(case_type, "判例")

    return "その他"


def short_preview(text: str, max_chars: int = 120) -> str:
    """One-line preview text suitable for plotly hover tooltips."""
    if not text:
        return ""
    cleaned = text.replace("\n", " ").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "…"


# ---------------------------------------------------------------------------
# Disk Cache for Projections
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    path = Path(CHROMA_DB_PATH).parent / ".viz_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _embedding_fingerprint(embeddings: np.ndarray) -> str:
    """Stable hash of an embedding matrix shape + first/last rows."""
    contig = np.ascontiguousarray(embeddings, dtype=np.float32)
    head = contig[:1].tobytes() if contig.shape[0] else b""
    tail = contig[-1:].tobytes() if contig.shape[0] else b""
    shape_bytes = str(contig.shape).encode()
    h = hashlib.sha1()
    h.update(shape_bytes)
    h.update(head)
    h.update(tail)
    return h.hexdigest()[:16]


def project_cached(
    embeddings: np.ndarray,
    *,
    n_components: int,
    method: str,
) -> np.ndarray:
    """Project embeddings, caching the result on disk.

    Key: ``(method, n_components, fingerprint(embeddings))``.

    The disk cache lives next to the ChromaDB directory so a single
    ingestion can warm both stores.  Returns the cached array on hit and
    the freshly computed array on miss.
    """
    if embeddings.shape[0] == 0:
        return np.empty((0, n_components), dtype=np.float32)

    fp = _embedding_fingerprint(embeddings)
    path = _cache_dir() / f"proj_{method}_{n_components}d_{fp}.npy"
    if path.exists():
        try:
            coords = np.load(path)
            if coords.shape == (embeddings.shape[0], n_components):
                return coords.astype(np.float32)
        except Exception:
            logger.exception("射影キャッシュの読み込みに失敗: %s", path)

    coords = _project(
        embeddings,
        n_components=n_components,
        method=method,
        random_state=42,
        perplexity=30.0,
    )
    try:
        np.save(path, coords)
    except Exception:
        logger.exception("射影キャッシュの保存に失敗: %s", path)
    return coords


def precompute_projections(*, methods: tuple[str, ...] = ("pca", "tsne")) -> None:
    """Warm the disk cache for every method × dimension combination.

    Call this once after :func:`rag_system.ingest.run_ingestion` to make
    the Streamlit page snappy on first load.
    """
    data = fetch_all_embeddings()
    if data["embeddings"].shape[0] == 0:
        logger.warning("コーパスが空です。先に取り込みを実行してください。")
        return
    for method in methods:
        for n in (2, 3):
            logger.info("射影をプリ計算: method=%s n_components=%d", method, n)
            project_cached(data["embeddings"], n_components=n, method=method)
    logger.info("射影プリ計算完了")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_stats() -> None:
    stats = get_collection_stats()
    print("=== 書庫統計 ===")
    print(f"総チャンク数: {stats['total']:,}")
    print(f"ベクトル次元: {stats['dimension']}")
    print("\n文書タイプ別:")
    for k, v in stats["by_document_type"].items():
        print(f"  {k}: {v:,}")
    print("\n事件種別:")
    for k, v in stats["by_case_type"].items():
        print(f"  {k}: {v:,}")


def _cli_search(query: str, k: int) -> None:
    print(f"クエリ: {query!r}")
    print(f"取得件数: {k}")
    print()
    data = fetch_all_embeddings()
    query_emb = embed_texts([query])[0]
    idx, scores = top_k_similar(query_emb, data["embeddings"], k=k)
    for rank, (i, score) in enumerate(zip(idx, scores), 1):
        meta = data["metadatas"][i]
        label = group_label(meta)
        source = meta.get("source", "")
        case_id = meta.get("case_id", "")
        head = f"#{rank} [類似度 {score:.4f}] [{label}]"
        if case_id:
            head += f" {case_id}"
        if source:
            head += f" — {source}"
        print(head)
        preview = short_preview(data["documents"][i], 200)
        print(f"  {preview}")
        print()


def _cli_compare(text_a: str, text_b: str) -> None:
    emb = embed_texts([text_a, text_b])
    sim = cosine_similarity(emb[0], emb[1])
    print(f"A: {text_a[:80]}…" if len(text_a) > 80 else f"A: {text_a}")
    print(f"B: {text_b[:80]}…" if len(text_b) > 80 else f"B: {text_b}")
    print()
    print(f"コサイン類似度: {sim:.4f}")
    if sim >= 0.85:
        print("→ 非常に類似（ほぼ同義）")
    elif sim >= 0.70:
        print("→ 強く類似（同じトピックの別表現）")
    elif sim >= 0.50:
        print("→ やや類似（共通要素あり、別物）")
    elif sim >= 0.30:
        print("→ 弱い類似（ジャンルや雰囲気だけ共通）")
    else:
        print("→ ほぼ無関係")


def _cli_benchmark() -> None:
    """Measure how long each piece of the vector pipeline takes."""
    import time

    print("=== ベクトル操作ベンチマーク ===\n")

    # 1. Embedding model load
    t0 = time.perf_counter()
    get_embedding_model()
    print(f"1. 埋め込みモデルロード     : {time.perf_counter() - t0:6.2f} 秒")

    # 2. ChromaDB fetch
    t0 = time.perf_counter()
    data = fetch_all_embeddings()
    n = data["embeddings"].shape[0]
    print(f"2. ChromaDB 全件取得 ({n}件) : {time.perf_counter() - t0:6.2f} 秒")

    # 3. Embed one query
    t0 = time.perf_counter()
    q = embed_texts(["窃盗罪の量刑基準を示せ"])
    print(f"3. クエリ埋め込み (1 件)    : {time.perf_counter() - t0:6.2f} 秒")

    # 4. Top-K search
    t0 = time.perf_counter()
    top_k_similar(q[0], data["embeddings"], k=10)
    print(f"4. 上位 10 件検索           : {time.perf_counter() - t0:6.2f} 秒")

    # 5. Cosine similarity matrix
    t0 = time.perf_counter()
    cosine_similarity_matrix(data["embeddings"][:200])
    print(f"5. 200x200 類似度行列計算   : {time.perf_counter() - t0:6.2f} 秒")

    # 6. PCA 2D (cold)
    import shutil
    cache = _cache_dir()
    if cache.exists():
        shutil.rmtree(cache)
    t0 = time.perf_counter()
    project_cached(data["embeddings"], n_components=2, method="pca")
    print(f"6. PCA 2D (キャッシュ無し)  : {time.perf_counter() - t0:6.2f} 秒")

    # 7. PCA 2D (warm)
    t0 = time.perf_counter()
    project_cached(data["embeddings"], n_components=2, method="pca")
    print(f"7. PCA 2D (キャッシュ有り)  : {time.perf_counter() - t0:6.2f} 秒")

    # 8. t-SNE 2D (cold)
    t0 = time.perf_counter()
    project_cached(data["embeddings"], n_components=2, method="tsne")
    print(f"8. t-SNE 2D (キャッシュ無し): {time.perf_counter() - t0:6.2f} 秒")

    print(
        "\n→ Streamlit の初回ロードは「2 + 6 + 8」の合計が支配的。"
        "\n  `python -m rag_system.vector_analysis precompute` でキャッシュを作れば"
        "\n  以降は「2 + 7」（数百ミリ秒）で起動できます。"
    )


def _cli_story_demo() -> None:
    """Run the 桃太郎/浦島太郎/三匹の子豚 demo from the terminal."""
    titles = [
        "桃太郎", "浦島太郎", "三匹の子豚", "シンデレラ",
        "謎の国家・刑法", "謎の国家・憲法",
    ]
    texts = [
        "桃から生まれた男の子が動物を連れて鬼ヶ島へ鬼退治に行く昔話。",
        "亀を助けた青年が竜宮城に招かれ、長い時を過ごした後に老人になる昔話。",
        "わら・木・レンガで家を建てた三兄弟と、それを襲うオオカミの物語。",
        "継母にいじめられた娘が魔法で舞踏会へ行き、ガラスの靴で王子に見つけられる童話。",
        "謎の国家の刑法は犯罪の構成要件と量刑を定める基本法である。",
        "謎の国家憲法は基本的人権・統治機構・法の支配を定める最高法規である。",
    ]
    embeddings = embed_texts(texts)
    sim = cosine_similarity_matrix(embeddings)
    print("=== 物語ベクトル類似度マトリックス ===")
    print()
    header = "          " + "  ".join(f"{t:>8s}" for t in titles)
    print(header)
    for i, t in enumerate(titles):
        row = f"{t:>8s}  " + "  ".join(
            f"{sim[i, j]:8.3f}" for j in range(len(titles))
        )
        print(row)


def main() -> None:
    """Command-line entry point for ad-hoc vector analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="謎の国家 - ベクトル空間ツール",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="書庫の統計情報を表示")

    p_search = sub.add_parser("search", help="クエリで近傍検索")
    p_search.add_argument("query", type=str, help="検索クエリ")
    p_search.add_argument(
        "-k", "--k", type=int, default=5, help="取得件数 (デフォルト: 5)"
    )

    p_compare = sub.add_parser(
        "compare", help="二つのテキストのコサイン類似度を計算"
    )
    p_compare.add_argument("text_a", type=str, help="文書 A")
    p_compare.add_argument("text_b", type=str, help="文書 B")

    sub.add_parser(
        "story", help="桃太郎・浦島太郎・三匹の子豚などのデモを表示",
    )
    sub.add_parser(
        "precompute",
        help="2D/3D 射影をディスクにキャッシュ（Streamlit の初回起動を高速化）",
    )
    sub.add_parser(
        "benchmark",
        help="埋め込み・検索・次元削減の所要時間を計測",
    )

    args = parser.parse_args()
    if args.cmd == "stats":
        _cli_stats()
    elif args.cmd == "search":
        _cli_search(args.query, args.k)
    elif args.cmd == "compare":
        _cli_compare(args.text_a, args.text_b)
    elif args.cmd == "story":
        _cli_story_demo()
    elif args.cmd == "precompute":
        precompute_projections()
    elif args.cmd == "benchmark":
        _cli_benchmark()


if __name__ == "__main__":
    main()
