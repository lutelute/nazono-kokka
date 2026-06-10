"""類似度＆解析ツール ― 単体起動エントリ。

    streamlit run highlight_tool.py      (または ./run_tool.sh)

本体は ui/analysis_page.py。同じ画面は app.py（メインUI）の
「類似度＆解析」ページからも使えます。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from ui.analysis_page import render_analysis_page

st.set_page_config(page_title="謎の国家 RAG ― 類似度＆解析ツール",
                   page_icon="🔬", layout="wide")

render_analysis_page()
