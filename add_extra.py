"""rag_system/extra_corpus.py の物語・法令を ChromaDB に追加（upsert）。

実行:
    python add_extra.py

既存の書庫（謎の国家の法令・判例）に、解析を豊かにするための物語と追加法令を
混ぜる。``upsert`` なので何度実行しても重複しない。物語は ``document_type="story"``、
``genre``（日本昔話/世界童話）を持つので、書庫マップ・クラスタ発見で独自の群として現れる。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    import chromadb

    from rag_system.config import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH
    from rag_system.extra_corpus import LAWS, STORIES
    from rag_system.vector_analysis import embed_texts

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    coll = client.get_collection(CHROMA_COLLECTION_NAME)
    before = coll.count()

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    for name, d in STORIES.items():
        ids.append(f"story::{name}")
        docs.append(d["text"])
        metas.append({"document_type": "story", "genre": d["genre"],
                      "source": f"物語/{name}"})

    for name, d in LAWS.items():
        ids.append(f"law::{name}")
        docs.append(d["text"])
        metas.append({"document_type": "legal_framework",
                      "filename": d["filename"], "source": f"法令/{name}"})

    print(f"埋め込み計算中… ({len(docs)} 件)")
    embs = embed_texts(docs)

    coll.upsert(ids=ids, embeddings=embs.tolist(),
                documents=docs, metadatas=metas)

    after = coll.count()
    print(f"追加完了: 物語 {len(STORIES)} 話 + 法令 {len(LAWS)} 件 = {len(ids)} 件")
    print(f"書庫チャンク数: {before:,} → {after:,}")


if __name__ == "__main__":
    main()
