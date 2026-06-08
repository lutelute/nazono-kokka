"""Deepen the precedent corpus by expanding each case's legal reasoning.

The seed precedents carry only a 2-sentence ``reasoning`` field, so retrieved
context and generated answers look thin and many same-type cases read almost
identically. This script writes a richer, structured ``reasoning_detailed``
field built from each case's *own* metadata (charges, statutes, principles,
verdict, sentence), organised like a real judicial opinion:

    認定事実 → 適用法令とその趣旨 → 法的判断（あてはめ）→ 結論・量刑の理由

It is **non-destructive**: the original ``reasoning`` is preserved and a new
field is added. ``rag_system.ingest`` prefers ``reasoning_detailed`` when
present, so re-ingesting picks up the richer text automatically.

Two modes:
  * deterministic (default) — template-based, fast, covers all 1,206 cases.
  * ``--llm`` — rewrite via Ollama for the most natural prose (slow; use
    ``--limit`` to sample).

Usage::

    python -m scripts.enrich_precedents --dry-run --limit 3   # preview
    python -m scripts.enrich_precedents                        # all, deterministic
    python -m scripts.enrich_precedents --llm --limit 20       # 20 via Ollama
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_system.config import (
    PRECEDENTS_CIVIL_DIR,
    PRECEDENTS_CONSTITUTIONAL_DIR,
    PRECEDENTS_CRIMINAL_DIR,
)

_CATEGORY_DIRS = [
    ("criminal", PRECEDENTS_CRIMINAL_DIR),
    ("civil", PRECEDENTS_CIVIL_DIR),
    ("constitutional", PRECEDENTS_CONSTITUTIONAL_DIR),
]


# ---------------------------------------------------------------------------
# Deterministic, metadata-grounded reasoning builder
# ---------------------------------------------------------------------------


def _verdict_conclusion(category: str, verdict: str, sentence: str) -> str:
    """Phrase the conclusion section for the given verdict."""
    v = verdict or ""
    if "無罪" in v:
        return (
            f"以上の検討によれば、本件公訴事実については合理的な疑いを超える証明が"
            f"あるとはいえない。よって被告人は無罪である（{sentence or '無罪'}）。"
            "刑事裁判における無罪推定の原則および挙証責任の所在に照らした帰結である。"
        )
    if "違憲" in v:
        return (
            f"以上のとおり、本件で問題となった国家行為は憲法の保障する権利を"
            f"不当に制約するものであって違憲と解すべきである。救済として{sentence or '違憲確認'}を相当とする。"
            "立法・行政に対する違憲審査権の行使として、権利保障を実効的たらしめる判断である。"
        )
    if "合憲" in v:
        return (
            "以上の検討によれば、本件における権利の制約は正当な立法目的に基づき、"
            "目的と手段との間に合理的関連性が認められ、必要最小限度を逸脱しない。"
            f"したがって本件規制は合憲である（{sentence or '合憲'}）。"
        )
    if category == "civil":
        if "認容" in v or "一部認容" in v:
            return (
                f"以上によれば、原告の請求は理由がある。よって{sentence or '請求を認容する'}。"
                "当事者間の権利義務関係および損害の公平な分担の見地から導かれる結論である。"
            )
        if "棄却" in v:
            return (
                f"以上によれば、原告の請求は理由がない。よって{sentence or '請求を棄却する'}。"
                "要件事実についての主張立証が尽くされていないことによる帰結である。"
            )
    # default: 有罪 / 一部有罪 等
    return (
        f"以上の事情を総合的に考慮し、主文のとおり{sentence or '量刑を決定した'}。"
        "犯行の動機・態様の悪質性、結果の重大性、被害回復の有無、"
        "被告人の反省の情その他一切の情状を斟酌した結果である。"
    )


def build_detailed_reasoning(data: dict, category: str) -> str:
    """Compose a structured, case-specific reasoning narrative."""
    summary = (data.get("summary") or "").strip()
    reasoning = (data.get("reasoning") or "").strip()
    charges = data.get("charges") or []
    statutes = data.get("referenced_statutes") or []
    principles = data.get("legal_principles") or []
    verdict = data.get("verdict") or ""
    sentence = data.get("sentence") or ""

    parts: list[str] = []

    # 1. 認定事実
    facts = summary or reasoning
    charge_line = ""
    if charges:
        charge_line = "本件で審理の対象となった訴因は、" + "、".join(charges) + "である。"
    parts.append(
        "【認定事実】\n"
        + facts
        + (("\n" + charge_line) if charge_line else "")
    )

    # 2. 適用法令とその趣旨
    if statutes:
        primary = statutes[0]
        others = "、".join(statutes[1:]) if len(statutes) > 1 else ""
        purpose = principles[0] if principles else "本件法益の保護"
        law_text = (
            f"本件の判断において中心となる規定は{primary}である。"
            f"同条は、{purpose}を趣旨とするものであり、"
            "その文言および立法目的に照らして要件を解釈する必要がある。"
        )
        if others:
            law_text += f"あわせて{others}の適用関係についても検討を要する。"
        parts.append("【適用法令とその趣旨】\n" + law_text)

    # 3. 法的判断（あてはめ）
    if principles:
        principle_text = (
            "本件では、" + "、".join(principles[:3]) + "が主たる争点となる。"
            "認定した事実をこれらの法的観点に照らして検討すると、"
            "被告人（被告）の行為態様、主観的要素および結果との因果関係について、"
            "前記法令の定める要件への当てはめを慎重に行うべきである。"
        )
        parts.append("【法的判断】\n" + principle_text)

    # 4. 結論・量刑の理由
    parts.append("【結論】\n" + _verdict_conclusion(category, verdict, sentence))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Optional LLM rewrite (Ollama)
# ---------------------------------------------------------------------------


def llm_reasoning(data: dict, category: str) -> str | None:
    """Rewrite reasoning via Ollama; return None on any failure."""
    import json as _json
    import urllib.error
    import urllib.request

    from rag_system.config import LLM_MODEL_NAME, OLLAMA_BASE_URL

    prompt = (
        "あなたは謎の国家の裁判官です。以下の判例メタデータに基づき、"
        "判決理由を法的に丁寧な日本語で15〜20文程度に拡張してください。"
        "認定事実・適用法令とその趣旨・法的判断（あてはめ）・結論の順で構成してください。"
        "メタデータにない事実を創作しすぎないこと。\n\n"
        f"事件種別: {data.get('case_type')}\n"
        f"罪状: {data.get('charges')}\n"
        f"参照法令: {data.get('referenced_statutes')}\n"
        f"法的原則: {data.get('legal_principles')}\n"
        f"判決: {data.get('verdict')} / {data.get('sentence')}\n"
        f"概要: {data.get('summary')}\n"
        f"元の判決理由: {data.get('reasoning')}\n"
    )
    payload = {
        "model": LLM_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3},
    }
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = _json.loads(resp.read().decode("utf-8"))
            text = (out.get("response") or "").strip()
            return text or None
    except (urllib.error.URLError, OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def iter_precedent_files():
    for category, dir_path in _CATEGORY_DIRS:
        if not dir_path.exists():
            continue
        for jp in sorted(dir_path.glob("*.json")):
            yield category, jp


def main() -> None:
    parser = argparse.ArgumentParser(description="判例の判決理由を拡充")
    parser.add_argument("--llm", action="store_true",
                        help="Ollama で判決理由を書き換える（遅い）")
    parser.add_argument("--limit", type=int, default=0,
                        help="処理件数の上限（0 = 全件）")
    parser.add_argument("--dry-run", action="store_true",
                        help="ファイルを書き換えず、生成結果を表示")
    parser.add_argument("--force", action="store_true",
                        help="既に reasoning_detailed があっても再生成")
    args = parser.parse_args()

    processed = 0
    skipped = 0
    for category, jp in iter_precedent_files():
        if args.limit and processed >= args.limit:
            break
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if data.get("reasoning_detailed") and not args.force:
            skipped += 1
            continue

        detailed = None
        if args.llm:
            detailed = llm_reasoning(data, category)
        if not detailed:
            detailed = build_detailed_reasoning(data, category)

        if args.dry_run:
            print("=" * 70)
            print(f"{jp.name}  ({category} / {data.get('verdict')})")
            print(detailed)
            processed += 1
            continue

        data["reasoning_detailed"] = detailed
        jp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        processed += 1
        if processed % 100 == 0:
            print(f"  ... {processed} 件処理")

    mode = "LLM" if args.llm else "決定論"
    action = "プレビュー" if args.dry_run else "書き込み"
    print(f"完了（{mode}・{action}）: {processed} 件処理, {skipped} 件スキップ")
    if not args.dry_run and processed:
        print("再取り込みで反映されます: python -m rag_system.ingest --reset")


if __name__ == "__main__":
    main()
