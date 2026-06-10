"""Smoke test: ``render_similarity_highlight`` が Streamlit 上で描画できるか。

``AppTest`` で 3 タブを実際に描画し、未捕捉例外が無いこと、ハイライト用の
``rgba`` 背景がマークダウンに現れることを確認する。モデルと streamlit が要るので
無ければ skip。
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("sentence_transformers")
pytest.importorskip("torch")

from streamlit.testing.v1 import AppTest  # noqa: E402


def _smoke() -> None:
    from ui.highlight_view import render_similarity_highlight

    render_similarity_highlight(
        "桃から生まれた男の子が犬や猿を連れて鬼ヶ島へ鬼退治に行く。",
        "亀を助けた若い漁師が竜宮城へ招かれ長い時を過ごす。",
        label_a="桃太郎",
        label_b="浦島太郎",
        key_prefix="smoke",
    )


def test_render_raises_no_exception() -> None:
    at = AppTest.from_function(_smoke)
    at.run(timeout=120)
    assert not at.exception, f"描画で例外が発生: {at.exception}"


def test_render_emits_highlight_markup() -> None:
    at = AppTest.from_function(_smoke)
    at.run(timeout=120)
    blob = " ".join(m.value for m in at.markdown)
    assert "rgba(" in blob, "ハイライト背景(rgba)がマークダウンに出ていない"
