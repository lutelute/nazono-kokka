#!/usr/bin/env bash
# 謎の国家 RAG ― 類似度＆解析ツール 起動スクリプト
#
#   使い方:  ./run_tool.sh
#
# ブラウザが自動で開き、6タブのツールが使えます。
# （初回は埋め込みモデルのロードに少し時間がかかります）
set -e
cd "$(dirname "$0")"

if [ -x .venv/bin/streamlit ]; then
  exec .venv/bin/streamlit run highlight_tool.py
elif command -v streamlit >/dev/null 2>&1; then
  exec streamlit run highlight_tool.py
else
  echo "streamlit が見つかりません。先に依存をインストールしてください:"
  echo "  pip install -r requirements.txt"
  exit 1
fi
