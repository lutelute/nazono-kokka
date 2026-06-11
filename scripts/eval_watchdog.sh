#!/bin/bash
# 評価のお守りスクリプト — Ollama「Stopping...」スタック対策
#
# run_evaluation.py を --resume で起動し、incremental JSONL の更新が
# STALL_SEC 秒止まったら evalプロセスをkill → Ollama再起動 → --resume 再投入する。
# JSONL が TARGET 行に達したら終了する。
#
# 使い方: bash scripts/eval_watchdog.sh <retriever> <jsonl> <logfile> [target]
set -u
cd "$(dirname "$0")/.."

RETRIEVER="${1:?retriever (dense/advanced)}"
JSONL="${2:?incremental jsonl path}"
LOGFILE="${3:?eval log path}"
TARGET="${4:-139}"
STALL_SEC=600
WD_LOG="logs/eval_watchdog.log"

note() { echo "$(date '+%F %T') [watchdog] $*" | tee -a "$WD_LOG"; }

launch() {
  .venv/bin/python scripts/run_evaluation.py --cases all \
    --model qwen2.5:7b-instruct --num-ctx 8192 \
    --retriever "$RETRIEVER" --resume >> "$LOGFILE" 2>&1 &
  note "eval起動 (pid=$!, retriever=$RETRIEVER)"
}

count() { [ -f "$JSONL" ] && wc -l < "$JSONL" | tr -d ' ' || echo 0; }

note "開始: $(count)/$TARGET 件"
launch

while [ "$(count)" -lt "$TARGET" ]; do
  sleep 120
  n=$(count)
  if [ "$n" -ge "$TARGET" ]; then break; fi
  if ! pgrep -f "run_evaluation.py.*$RETRIEVER" >/dev/null; then
    note "evalプロセス消失 ($n/$TARGET) — 再起動"
    launch
    continue
  fi
  age=$(( $(date +%s) - $(stat -f %m "$JSONL") ))
  if [ "$age" -gt "$STALL_SEC" ]; then
    note "ストール検知 (${age}s 更新なし, $n/$TARGET) — eval kill + Ollama再起動"
    pkill -f "run_evaluation.py.*$RETRIEVER"
    sleep 3
    brew services restart ollama
    sleep 10
    launch
  fi
done

note "完走: $(count)/$TARGET 件"
