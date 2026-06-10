"""Tests for the pure helpers in :mod:`ui.highlight_view`.

色変換と HTML 組み立ては Streamlit 実行コンテキスト不要の純粋関数なので直接検証する。
主眼: (1) 色フォーマット、(2) ユーザー入力（単語）の HTML エスケープ、
(3) 寄与の符号と暖色/寒色の対応。``streamlit`` 未導入なら skip。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("streamlit")

from ui.highlight_view import (  # noqa: E402
    _PAIR_PALETTE,
    _heat_color,
    _heatmap_html,
    _hex_to_rgb,
    _pairing_html,
    _rgba,
    _sentence_block,
)


class TestColorHelpers:
    def test_hex_to_rgb(self) -> None:
        assert _hex_to_rgb("#ff0000") == (255, 0, 0)
        assert _hex_to_rgb("#1f77b4") == (31, 119, 180)

    def test_rgba_format(self) -> None:
        assert _rgba("#ff0000", 0.5) == "rgba(255, 0, 0, 0.500)"

    def test_palette_has_ten_colors(self) -> None:
        assert len(_PAIR_PALETTE) == 10
        assert all(c.startswith("#") and len(c) == 7 for c in _PAIR_PALETTE)

    def test_heat_color_positive_is_warm(self) -> None:
        assert "229" in _heat_color(1.0, 1.0)

    def test_heat_color_negative_is_cool(self) -> None:
        assert _heat_color(-1.0, 1.0).startswith("rgba(30,")

    def test_heat_color_zero_is_min_alpha(self) -> None:
        assert "0.100" in _heat_color(0.0, 1.0)

    def test_heat_color_clamps_alpha(self) -> None:
        c = _heat_color(10.0, 1.0)
        alpha = float(c.rsplit(",", 1)[1].strip().rstrip(")"))
        assert alpha <= 1.0


class TestHtmlBuilders:
    def test_heatmap_escapes_html(self) -> None:
        out = _heatmap_html(["<script>"], np.array([0.5]), np.array([False]), 1.0)
        assert "&lt;script&gt;" in out
        assert "<script>" not in out

    def test_heatmap_punct_is_grey(self) -> None:
        out = _heatmap_html(["。"], np.array([0.5]), np.array([True]), 1.0)
        assert "#c8c8c8" in out

    def test_heatmap_wraps_in_div(self) -> None:
        out = _heatmap_html(["桃"], np.array([0.3]), np.array([False]), 1.0)
        assert out.startswith("<div") and out.endswith("</div>")
        assert "桃" in out

    def test_sentence_block_colored_has_border(self) -> None:
        h = _sentence_block("ある文。", "#1f77b4", "→B文1 (0.73)")
        assert "ある文。" in h
        assert "border-left" in h
        assert "31, 119, 180" in h

    def test_sentence_block_none_is_grey(self) -> None:
        h = _sentence_block("対応の弱い文。", None, "対応弱")
        assert "対応の弱い文。" in h
        assert "#cccccc" in h

    def test_sentence_block_escapes(self) -> None:
        h = _sentence_block("<b>x</b>", "#1f77b4", "badge")
        assert "&lt;b&gt;" in h
        assert "<b>x</b>" not in h

    def test_pairing_html_adds_superscript(self) -> None:
        out = _pairing_html(["語", "他"], {0: 0}, {0: 1})
        assert "<sup" in out
        assert "語" in out and "他" in out

    def test_pairing_html_uncolored_is_plain(self) -> None:
        out = _pairing_html(["無関係"], {}, {})
        assert "無関係" in out
        assert "<sup" not in out

    def test_pairing_html_escapes(self) -> None:
        out = _pairing_html(["<i>", "x"], {0: 0}, {0: 1})
        assert "&lt;i&gt;" in out
        assert "<i>" not in out
