"""GitHub Pages 体験版（stlite）用の事前計算データを書き出す。

ブラウザ（Pyodide）では torch / sentence-transformers が動かないため、
埋め込み計算はここで済ませ、ブラウザ側は保存済みベクトルの numpy 演算
（コサイン類似度・文アライメント・散布図描画）だけを行う。

出力: docs/tool/data.json — stlite の files URL 指定で仮想 FS に配置される
（stlite はWeb Worker内でPythonを実行するため window 経由の受け渡しは不可）

Usage:
    .venv/bin/python scripts/export_web_demo_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from rag_system.config import EMBEDDINGS_MODEL_NAME
from rag_system.text_alignment import split_sentences, token_contributions, token_pairing
from ui.analysis_page import _PRESETS

OUT_PATH = PROJECT_ROOT / "docs" / "tool" / "data.json"


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def rounded(arr: np.ndarray, digits: int = 4) -> list:
    return np.round(arr.astype(float), digits).tolist()


def export_presets(model: SentenceTransformer) -> dict:
    """プリセット各話の全体ベクトルと文ベクトル。"""
    stories = {}
    for name, text in _PRESETS.items():
        if not text:
            continue
        sentences = split_sentences(text)
        sent_emb = model.encode(sentences, normalize_embeddings=True)
        whole_emb = model.encode([text], normalize_embeddings=True)[0]
        stories[name] = {
            "text": text,
            "sentences": sentences,
            "sent_vecs": rounded(np.asarray(sent_emb)),
            "vec": rounded(np.asarray(whole_emb)),
        }
        print(f"  {name}: {len(sentences)} 文")
    return stories


def export_pair_analyses() -> dict:
    """全プリセットペアのトークン寄与（厳密分解）とペアリング（max-sim）を事前計算。

    ブラウザでは torch が動かないため、語単位の埋め込みを要する計算は
    ここで済ませ、語と色値（centered）・対応 index だけを渡す。
    """
    names = [n for n, t in _PRESETS.items() if t]
    pairs: dict[str, dict] = {}
    for i, na in enumerate(names):
        for nb in names[i + 1 :]:
            tc = token_contributions(_PRESETS[na], _PRESETS[nb])
            tp = token_pairing(_PRESETS[na], _PRESETS[nb])

            def side(s):
                return {
                    "words": s.words,
                    "centered": rounded(s.centered),
                    "raw": rounded(s.raw, 5),
                    "punct": [bool(x) for x in s.is_punct],
                }

            pairs[f"{na}|{nb}"] = {
                "cosine": round(tc.cosine, 4),
                "a": side(tc.side_a),
                "b": side(tc.side_b),
                "a_to_b": [int(x) for x in tp.a_to_b],
                "a_to_b_score": rounded(tp.a_to_b_score),
                "b_to_a": [int(x) for x in tp.b_to_a],
                "b_to_a_score": rounded(tp.b_to_a_score),
            }
            print(f"  ペア解析: {na} × {nb} (cos={tc.cosine:.3f})")
    return pairs


def label_of(meta: dict) -> tuple[str, str]:
    """(グループ名, 大分類) を返す。大分類は legal / precedent / story。"""
    src = str(meta.get("source", ""))
    doc_type = meta.get("document_type", "")
    if doc_type == "story":
        return "物語(昔話など)", "story"
    if doc_type == "legal_framework" or "legal_framework" in src:
        name = Path(src).stem
        names = {
            "constitution": "憲法",
            "criminal_code": "刑法",
            "civil_code": "民法",
            "administrative_code": "行政法",
            "ethical_guidelines": "倫理指針",
            "cultural_regulations": "文化規制",
        }
        return names.get(name, name or "法令"), "legal"
    case_type = meta.get("case_type", "")
    names = {"criminal": "判例(刑事)", "civil": "判例(民事)", "constitutional": "判例(憲法)"}
    return names.get(case_type, "判例"), "precedent"


def doc_name(meta: dict) -> str:
    """近傍・重複表示用の短い文書名。"""
    m = meta or {}
    if m.get("case_id"):
        return str(m["case_id"])
    src = str(m.get("source", ""))
    return Path(src).name or "?"


def export_corpus(story_vecs: dict[str, list[float]]) -> dict:
    """ChromaDB の全文書埋め込みから地図・軸・近傍・クラスタ・重複データを作る。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_db"))
    col = client.get_collection("nazono_kokka_legal")
    got = col.get(include=["embeddings", "metadatas", "documents"])
    emb = normalize(np.asarray(got["embeddings"], dtype=np.float64))
    metas = got["metadatas"]
    docs = got["documents"]
    print(f"  コーパス: {emb.shape[0]} 文書 × {emb.shape[1]} 次元")

    groups, kinds, names = [], [], []
    for m in metas:
        g, k = label_of(m or {})
        groups.append(g)
        kinds.append(k)
        names.append(doc_name(m))

    pca = PCA(n_components=20, random_state=0)
    coords20 = pca.fit_transform(emb)
    explained = pca.explained_variance_ratio_

    is_legal = np.array([k == "legal" for k in kinds])
    is_prec = np.array([k == "precedent" for k in kinds])
    axis = emb[is_legal].mean(axis=0) - emb[is_prec].mean(axis=0)
    axis = axis / np.linalg.norm(axis)
    proj = emb @ axis

    # --- プリセットをクエリにした近傍 top10（境界の探究） ---
    neighbors = {}
    for sname, svec in story_vecs.items():
        sims = emb @ np.asarray(svec)
        top = np.argsort(sims)[::-1][:10]
        neighbors[sname] = [
            {"group": groups[i], "name": names[i], "sim": round(float(sims[i]), 4)}
            for i in top
        ]
    print(f"  近傍検索: {len(neighbors)} プリセット × top10")

    # --- K-means（k は固定4種を事前計算、ブラウザで切替） ---
    kmeans_labels = {}
    for k in (4, 6, 8, 10):
        km = KMeans(n_clusters=k, n_init=4, random_state=0)
        kmeans_labels[str(k)] = [int(x) for x in km.fit_predict(emb)]
    print("  K-means: k=4,6,8,10")

    # --- 書庫内の重複候補 top30（自己と同一文書ペアの対角は除外） ---
    sim_mat = emb @ emb.T
    iu = np.triu_indices(len(emb), k=1)
    flat = sim_mat[iu]
    top_idx = np.argsort(flat)[::-1][:30]
    duplicates = [
        {
            "a": names[iu[0][t]], "ga": groups[iu[0][t]],
            "b": names[iu[1][t]], "gb": groups[iu[1][t]],
            "sim": round(float(flat[t]), 4),
            "head_a": (docs[iu[0][t]] or "")[:42],
            "head_b": (docs[iu[1][t]] or "")[:42],
        }
        for t in top_idx
    ]
    print("  重複検出: top30")

    return {
        "pca_xy": rounded(coords20[:, :2], 3),
        "pca_xyz": rounded(coords20[:, :3], 3),
        "explained_variance": rounded(explained, 4),
        "axis_projection": rounded(proj, 4),
        "group": groups,
        "neighbors": neighbors,
        "kmeans": kmeans_labels,
        "duplicates": duplicates,
    }


def main() -> int:
    print("埋め込みモデルをロード中…")
    model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)

    print("プリセット物語を埋め込み…")
    stories = export_presets(model)

    print("ペア解析（トークン寄与＋ペアリング）…")
    pairs = export_pair_analyses()

    print("コーパスを処理…")
    corpus = export_corpus({n: s["vec"] for n, s in stories.items()})

    data = {
        "model": EMBEDDINGS_MODEL_NAME,
        "stories": stories,
        "pairs": pairs,
        "corpus": corpus,
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(payload, encoding="utf-8")
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"出力: {OUT_PATH} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
