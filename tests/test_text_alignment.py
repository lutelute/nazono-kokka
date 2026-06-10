"""Tests for :mod:`rag_system.text_alignment`.

軽量な純粋関数（文分割・記号判定）はモデル無しで検証する。埋め込みモデルを
要する関数は ``sentence_transformers`` がある環境でのみ実行し、無ければ skip する。

主眼: (1) トークン寄与の総和がコサイン類似度に一致（厳密分解）、(2) 単語に
相手文の語が混入しない、(3) 同一テキスト同士は各語・各文が自分自身と対応する。

注: このモデルの tokenizer は日本語を細かく（「む/か/し」等）分割し、文単位の
意味類似もやや弱い。そのため意味的な大小関係に依存せず、恒等性（同一テキストの
自己対応）と構造（厳密分解・混入なし）でコアを検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_system.text_alignment import (
    is_punctuation,
    sentence_alignment,
    split_sentences,
    token_contributions,
    token_pairing,
    word_level,
)


class TestSplitSentences:
    def test_basic_two_sentences(self) -> None:
        assert split_sentences("これは一文。これは二文。") == [
            "これは一文。", "これは二文。"]

    def test_newline_splits(self) -> None:
        assert split_sentences("一行目\n二行目") == ["一行目", "二行目"]

    def test_question_and_exclamation(self) -> None:
        assert split_sentences("元気？はい！") == ["元気？", "はい！"]

    def test_empty_and_whitespace(self) -> None:
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []

    def test_no_terminator_returns_whole(self) -> None:
        assert split_sentences("句点なしの文") == ["句点なしの文"]


class TestIsPunctuation:
    @pytest.mark.parametrize("s", ["。", "、", "・", "！？", "…"])
    def test_punctuation_true(self, s: str) -> None:
        assert is_punctuation(s) is True

    @pytest.mark.parametrize("s", ["桃太郎", "A", "鬼", "太郎"])
    def test_word_false(self, s: str) -> None:
        assert is_punctuation(s) is False


# 厳密分解は特殊トークン分だけずれるため、十分長い本文で測る。
_TEXT_A = (
    "むかしむかし、おばあさんが川で洗濯をしていると大きな桃が流れてきた。"
    "桃を割ると元気な男の子が現れた。"
    "男の子は桃太郎と名付けられ、やがて鬼ヶ島へ鬼退治の旅に出た。"
    "イヌ・サル・キジを家来にして鬼を倒し、宝物を持ち帰った。"
)
_TEXT_B = (
    "むかしむかし、浦島太郎という若い漁師がいた。"
    "ある日、子供たちにいじめられていた亀を助けた。"
    "亀のお礼に竜宮城へ連れて行かれ、乙姫様と楽しく過ごした。"
    "玉手箱を開けると白い煙が出て老人になってしまった。"
)
_TEXT_C = "謎の国家の刑法第235条は他人の財物を窃取した者の窃盗罪を定める。"


@pytest.fixture(scope="module")
def _model() -> None:
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("torch")


@pytest.mark.usefixtures("_model")
class TestWordLevel:
    def test_words_and_counts(self) -> None:
        wl = word_level(_TEXT_A)
        assert len(wl.words) > 0
        assert wl.embeddings.shape[0] == len(wl.words)
        assert int(wl.token_counts.sum()) == len(wl.words)
        assert wl.n_tokens >= len(wl.words)

    def test_empty_text(self) -> None:
        wl = word_level("")
        assert wl.words == []
        assert wl.embeddings.shape[0] == 0
        assert wl.n_tokens == 0


@pytest.mark.usefixtures("_model")
class TestTokenContributions:
    def test_raw_sum_equals_cosine(self) -> None:
        # 表示語の raw 総和 ≈ cos（差は特殊トークン <s></s>▁ の寄与のみ）。
        # この多言語モデルは特殊トークンの寄与がやや大きめなので許容を 0.05 とする。
        tc = token_contributions(_TEXT_A, _TEXT_B)
        assert tc.side_a.raw.sum() == pytest.approx(tc.cosine, abs=0.05)
        assert tc.side_b.raw.sum() == pytest.approx(tc.cosine, abs=0.05)

    def test_no_cross_contamination(self) -> None:
        tc = token_contributions(_TEXT_A, _TEXT_B)
        haystack = _TEXT_A.replace("、", "").replace("。", "")
        for w in tc.side_a.words:
            if not is_punctuation(w):
                assert w in haystack, f"{w!r} は A 本文に無い（混入）"

    def test_centered_is_zero_mean(self) -> None:
        tc = token_contributions(_TEXT_A, _TEXT_B)
        assert float(tc.side_a.centered.mean()) == pytest.approx(0.0, abs=1e-5)

    def test_related_more_similar_than_unrelated(self) -> None:
        related = token_contributions(_TEXT_A, _TEXT_B).cosine
        unrelated = token_contributions(_TEXT_A, _TEXT_C).cosine
        assert related > unrelated


@pytest.mark.usefixtures("_model")
class TestSentenceAlignment:
    def test_matrix_shape(self) -> None:
        sa = sentence_alignment(_TEXT_A, _TEXT_B)
        assert sa.similarity.shape == (
            len(sa.sentences_a), len(sa.sentences_b))

    def test_argmax_in_bounds(self) -> None:
        sa = sentence_alignment(_TEXT_A, _TEXT_B)
        assert sa.a_to_b.shape[0] == len(sa.sentences_a)
        assert int(sa.a_to_b.max()) < len(sa.sentences_b)
        assert float(sa.a_to_b_score.max()) <= 1.0 + 1e-5

    def test_self_alignment_is_identity(self) -> None:
        # 同一テキスト同士なら各文が自分自身（対角）と最も対応する（恒等性）。
        sa = sentence_alignment(_TEXT_A, _TEXT_A)
        for i in range(len(sa.sentences_a)):
            assert int(sa.a_to_b[i]) == i
            assert float(sa.a_to_b_score[i]) > 0.99


@pytest.mark.usefixtures("_model")
class TestTokenPairing:
    def test_self_pairing_is_perfect(self) -> None:
        tp = token_pairing(_TEXT_A, _TEXT_A)
        non_punct = [
            float(s) for s, p in zip(tp.a_to_b_score, tp.is_punct_a) if not p]
        assert non_punct
        assert min(non_punct) > 0.99

    def test_similarity_shape(self) -> None:
        tp = token_pairing(_TEXT_A, _TEXT_B)
        assert tp.similarity.shape == (len(tp.words_a), len(tp.words_b))

    def test_scores_bounded(self) -> None:
        tp = token_pairing(_TEXT_A, _TEXT_B)
        assert float(tp.a_to_b_score.max()) <= 1.0 + 1e-5
        assert float(tp.a_to_b_score.min()) >= -1.0 - 1e-5

    def test_shared_word_pairs_strongly(self) -> None:
        # 「太郎」は両文書にあり tokenizer が1トークンにまとめる。相手の「太郎」と高類似。
        tp = token_pairing(_TEXT_A, _TEXT_B)
        idx = [i for i, w in enumerate(tp.words_a) if w == "太郎"]
        assert idx, "テスト前提: A に『太郎』トークンがある"
        for i in idx:
            assert float(tp.a_to_b_score[i]) > 0.5
