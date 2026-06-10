"""768 次元埋め込みから『意味のある軸』を抽出し、その分散を測る。

埋め込み空間は 768 次元あるが、情報は少数の方向に偏って乗っている。この
モジュールは「どの方向が意味を持つか」を 2 通りで取り出し、その軸の上で
文書がどう散らばるか（分散）を定量化する。

- :func:`explained_variance` ― PCA で分散最大の直交軸を順に取り、各軸が
  全分散の何 % を説明するかを返す（埋め込みの実効次元の指標）。
- :func:`concept_axis` ― 2 グループの重心差から解釈可能な対比軸を作る
  （例「法令 ⇄ 判例」「有罪 ⇄ 無罪」）。
- :func:`project_on_axis` / :func:`axis_variance_by_group` ― 抽出した軸へ
  文書を射影し、グループごとの平均・分散で「軸上の分布」を見る。

純粋 numpy（PCA のみ scikit-learn）で、埋め込みモデルには依存しない。
"""

from __future__ import annotations

import numpy as np


def explained_variance(
    embeddings: np.ndarray, n_components: int = 20
) -> dict[str, np.ndarray]:
    """PCA で上位主成分の説明分散比を返す。

    各主成分は分散が最大になる直交方向（＝最も『意味のある軸』）で、その
    固有値がその方向の分散。``cumulative`` を見れば「上位 k 軸で全分散の
    何 % を説明できるか」が分かり、768 次元の実効次元の目安になる。

    Returns
    -------
    dict
        ``ratio`` (各軸の説明分散比), ``cumulative`` (累積),
        ``variance`` (各軸の分散の絶対値), ``components`` (``(k, dim)`` 各軸の
        方向ベクトル)。
    """
    dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
    if embeddings.shape[0] < 2 or dim == 0:
        z = np.zeros(0, np.float32)
        return {"ratio": z, "cumulative": z, "variance": z,
                "components": np.zeros((0, dim), np.float32)}

    from sklearn.decomposition import PCA

    n = min(n_components, embeddings.shape[0], dim)
    pca = PCA(n_components=n, random_state=42)
    pca.fit(embeddings)
    ratio = pca.explained_variance_ratio_.astype(np.float32)
    return {
        "ratio": ratio,
        "cumulative": np.cumsum(ratio).astype(np.float32),
        "variance": pca.explained_variance_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
    }


def concept_axis(
    embeddings: np.ndarray,
    mask_positive: np.ndarray,
    mask_negative: np.ndarray,
) -> np.ndarray:
    """2 グループの重心差から『意味のある対比軸』を作る（単位ベクトル）。

    例: 有罪判例の重心 − 無罪判例の重心 → 「有罪 ⇄ 無罪」方向。これに文書を
    射影すると、その文書がどちら寄りかを 1 次元で読める。どちらかのグループが
    空なら零ベクトルを返す。
    """
    pos = embeddings[mask_positive]
    neg = embeddings[mask_negative]
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        return np.zeros(embeddings.shape[1], np.float32)
    axis = pos.mean(axis=0) - neg.mean(axis=0)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return axis.astype(np.float32)
    return (axis / norm).astype(np.float32)


def project_on_axis(embeddings: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """各文書ベクトルを軸（方向ベクトル）へ射影したスカラー値 ``(N,)``。"""
    if embeddings.size == 0:
        return np.zeros(0, np.float32)
    return (embeddings @ axis).astype(np.float32)


def axis_variance_by_group(
    projections: np.ndarray, labels: list[str]
) -> dict[str, dict[str, float]]:
    """軸への射影値をグループごとに平均・分散・件数で集計する。

    「抽出した軸の上で、各グループがどこに、どれだけ広がって分布するか」を
    返す。``mean`` が離れているほど軸がそのグループ対比をよく捉えており、
    ``std`` が小さいほどグループが軸上で締まっている（一貫している）。
    """
    arr = np.asarray(projections, dtype=np.float32)
    out: dict[str, dict[str, float]] = {}
    for g in sorted(set(labels)):
        mask = np.array([lbl == g for lbl in labels], dtype=bool)
        vals = arr[mask]
        if vals.size == 0:
            continue
        out[g] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "var": float(vals.var()),
            "count": int(vals.size),
        }
    return out
