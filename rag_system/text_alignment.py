"""二文書の「どこが似ているか」を抽出する類似度アトリビューション。

二つのテキストのコサイン類似度を *スカラー1個* から「文章のどの部分が似ているか」
へ分解する純粋計算ロジック（Streamlit 非依存）。

- :func:`sentence_alignment` ― 文・節レベルの対応付け（日本語で最も読みやすい）
- :func:`token_contributions` ― 単語レベルのヒートマップ（中心化コントラスト）
- :func:`token_pairing` ― 単語間 max-sim ペアリング（BERTScore 的）

埋め込みモデル（paraphrase-multilingual-mpnet-base-v2）は mean-pooling なので、
文ベクトル同士のコサイン類似度を各トークンの寄与の総和に厳密分解できる
（raw の総和＝コサイン類似度）。生の寄与は助詞が支配するため、中心化コントラスト
（全語平均を引く）で内容語を浮かせて色付けに使う。

トークン列は ``tokenize→forward`` で取得する（``encode`` 直呼びは列がズレる）。
SentencePiece のサブワードは「1トークン＝1表示単位」として扱う（日本語は語頭
マーカー ``▁`` がほぼ付かず、境界結合すると全体が1語に潰れるため）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from rag_system.vector_analysis import get_embedding_model

_SENT_SPLIT = re.compile(r"(?<=[。！？!?\n])")
_META_SPACE = "▁"
_SPECIAL_TOKENS = frozenset({"<s>", "</s>", "<pad>", "<unk>", "<mask>"})
_PUNCT_RE = re.compile(r"^[、。，．・…！？!?「」『』（）()【】\[\]〜~ーｰ\-—\s]+$")


def split_sentences(text: str) -> list[str]:
    """テキストを文（節）のリストに分割する。"""
    if not text or not text.strip():
        return []
    parts = _SENT_SPLIT.split(text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences or [text.strip()]


def is_punctuation(word: str) -> bool:
    """単語が句読点・記号のみで構成されているか。"""
    return bool(_PUNCT_RE.match(word))


@dataclass
class WordLevel:
    """単語単位に集約したトークン情報。

    Attributes
    ----------
    words: 表示用の単語（1トークン＝1単位）。
    embeddings: ``(W, dim)`` 各単語の埋め込み。
    token_counts: ``(W,)`` 各単語のサブワードトークン数（現状は全要素1）。
    sentence_vector: ``(dim,)`` 文ベクトル＝全トークン埋め込みの平均（mean-pooling）。
    n_tokens: 文ベクトルを作った有効トークン総数（特殊トークン込み、寄与の分母）。
    """

    words: list[str]
    embeddings: np.ndarray
    token_counts: np.ndarray
    sentence_vector: np.ndarray
    n_tokens: int


def word_level(text: str) -> WordLevel:
    """テキストを単語単位の埋め込みへ分解する（tokenize→forward）。"""
    import torch

    model = get_embedding_model()
    dim = int(model.get_sentence_embedding_dimension())

    if not text or not text.strip():
        return WordLevel([], np.zeros((0, dim), np.float32),
                         np.zeros(0, np.int64), np.zeros(dim, np.float32), 0)

    feats = model.tokenize([text])
    device = model.device
    feats = {k: v.to(device) for k, v in feats.items()}
    with torch.no_grad():
        out = model.forward(feats)

    emb = out["token_embeddings"][0].detach().cpu().numpy().astype(np.float32)
    ids = feats["input_ids"][0].detach().cpu().numpy()
    mask = feats["attention_mask"][0].detach().cpu().numpy().astype(bool)
    toks = model.tokenizer.convert_ids_to_tokens(ids)

    valid = emb[mask]
    sent_vec = (valid.mean(axis=0) if valid.shape[0]
                else np.zeros(dim, np.float32)).astype(np.float32)

    # 日本語では ▁ がほぼ付かないので 1トークン=1表示単位とする（特殊・空白は除外）
    words: list[str] = []
    word_embs: list[np.ndarray] = []
    for tok, vec, keep in zip(toks, emb, mask):
        if not keep or tok in _SPECIAL_TOKENS:
            continue
        piece = tok.replace(_META_SPACE, "")
        if not piece:
            continue
        words.append(piece)
        word_embs.append(vec)

    emb_arr = (np.vstack(word_embs).astype(np.float32)
               if word_embs else np.zeros((0, dim), np.float32))
    counts = np.ones(len(words), np.int64)
    return WordLevel(words, emb_arr, counts, sent_vec, int(mask.sum()))


@dataclass
class SentenceAlignment:
    sentences_a: list[str]
    sentences_b: list[str]
    similarity: np.ndarray
    a_to_b: np.ndarray
    a_to_b_score: np.ndarray
    b_to_a: np.ndarray
    b_to_a_score: np.ndarray


def sentence_alignment(text_a: str, text_b: str) -> SentenceAlignment:
    """文書 A・B を文に割り、文ペアのコサイン類似度と最良対応を返す。"""
    sa = split_sentences(text_a)
    sb = split_sentences(text_b)
    if not sa or not sb:
        empty = np.zeros((len(sa), len(sb)), np.float32)
        return SentenceAlignment(
            sa, sb, empty,
            np.zeros(len(sa), np.int64), np.zeros(len(sa), np.float32),
            np.zeros(len(sb), np.int64), np.zeros(len(sb), np.float32))

    model = get_embedding_model()
    ea = np.asarray(model.encode(sa, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False),
                    dtype=np.float32)
    eb = np.asarray(model.encode(sb, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False),
                    dtype=np.float32)
    sim = (ea @ eb.T).astype(np.float32)

    return SentenceAlignment(
        sa, sb, sim,
        sim.argmax(axis=1).astype(np.int64), sim.max(axis=1).astype(np.float32),
        sim.argmax(axis=0).astype(np.int64), sim.max(axis=0).astype(np.float32))


@dataclass
class SideContribution:
    words: list[str]
    raw: np.ndarray
    strength: np.ndarray
    centered: np.ndarray
    is_punct: np.ndarray


@dataclass
class TokenContributions:
    cosine: float
    side_a: SideContribution
    side_b: SideContribution


def _side_contribution(
    side: WordLevel, other_unit: np.ndarray, self_norm: float, total_tokens: int
) -> SideContribution:
    if side.embeddings.shape[0] == 0:
        z = np.zeros(0, np.float32)
        return SideContribution([], z, z, z, np.zeros(0, bool))
    raw = ((side.embeddings @ other_unit) * side.token_counts
           / (max(total_tokens, 1) * (self_norm or 1.0))).astype(np.float32)
    strength = (side.embeddings @ other_unit).astype(np.float32)
    centered = (strength - float(strength.mean())).astype(np.float32)
    is_punct = np.array([is_punctuation(w) for w in side.words], dtype=bool)
    return SideContribution(side.words, raw, strength, centered, is_punct)


def token_contributions(text_a: str, text_b: str) -> TokenContributions:
    """各単語がコサイン類似度にどれだけ効いているかを算出する。

    ``side_*.raw`` の総和は ``cosine`` にほぼ一致（厳密分解、差は特殊トークン分）。
    色付けには機能語を抑えた ``centered`` を使う。
    """
    wa = word_level(text_a)
    wb = word_level(text_b)
    a, b = wa.sentence_vector, wb.sentence_vector
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    a_unit = a / (na or 1.0)
    b_unit = b / (nb or 1.0)
    cosine = float(a_unit @ b_unit)
    side_a = _side_contribution(wa, b_unit, na, wa.n_tokens)
    side_b = _side_contribution(wb, a_unit, nb, wb.n_tokens)
    return TokenContributions(cosine, side_a, side_b)


@dataclass
class TokenPairing:
    words_a: list[str]
    words_b: list[str]
    similarity: np.ndarray
    a_to_b: np.ndarray
    a_to_b_score: np.ndarray
    b_to_a: np.ndarray
    b_to_a_score: np.ndarray
    is_punct_a: np.ndarray
    is_punct_b: np.ndarray


def token_pairing(text_a: str, text_b: str) -> TokenPairing:
    """文書 A の各単語と、文書 B で最も意味の近い単語を結ぶ (max-sim)。"""
    wa = word_level(text_a)
    wb = word_level(text_b)

    if wa.embeddings.shape[0] == 0 or wb.embeddings.shape[0] == 0:
        return TokenPairing(
            wa.words, wb.words,
            np.zeros((len(wa.words), len(wb.words)), np.float32),
            np.zeros(len(wa.words), np.int64), np.zeros(len(wa.words), np.float32),
            np.zeros(len(wb.words), np.int64), np.zeros(len(wb.words), np.float32),
            np.array([is_punctuation(w) for w in wa.words], bool),
            np.array([is_punctuation(w) for w in wb.words], bool))

    a_unit = wa.embeddings / (np.linalg.norm(wa.embeddings, axis=1, keepdims=True) + 1e-9)
    b_unit = wb.embeddings / (np.linalg.norm(wb.embeddings, axis=1, keepdims=True) + 1e-9)
    sim = (a_unit @ b_unit.T).astype(np.float32)

    return TokenPairing(
        wa.words, wb.words, sim,
        sim.argmax(axis=1).astype(np.int64), sim.max(axis=1).astype(np.float32),
        sim.argmax(axis=0).astype(np.int64), sim.max(axis=0).astype(np.float32),
        np.array([is_punctuation(w) for w in wa.words], bool),
        np.array([is_punctuation(w) for w in wb.words], bool))
