"""テストケースの期待条文を「謎の国家」法体系の実在条文に修正するパッチ。

基本セット 12 件のうち 10 件が、実在日本法の条番号（刑法第246条＝詐欺、
民法第709条＝不法行為、憲法第21条＝表現の自由など）を期待しており、
生成時に現実の法律知識が混入したと思われる。2パターンある：

1. コーパスのどこにも出現しない条番号（7件）→ statute_score が原理的に満点不可
2. コーパスに存在するが**意味が違う**条番号（3件）→ 正しく転記した回答が減点される
   （例: TC-011 期待「憲法第81条」は謎の国家では地方公共団体の権能。違憲審査は第71条。
   LLM は資料から第71条を正しく引用していたのにスコア 0.42 だった）

謎の国家の法令見出し・判例・GEN系ケースとの整合を突合して確定した対応：

| ケース  | 旧（実在日本法）         | 新（謎の国家）                        |
|---------|--------------------------|---------------------------------------|
| TC-002  | 刑法第199条（殺人）      | 刑法第151条（殺人）                   |
| TC-003  | 刑法第246条（詐欺）      | 刑法第198条（詐欺）                   |
| TC-004  | 刑法第36条（正当防衛）   | 刑法第22条（正当防衛）                |
| TC-005  | 民法第709条（不法行為）  | 民法第186条（不法行為による損害賠償） |
| TC-006  | 民法第570条（瑕疵担保）  | 民法第147条（売主の契約不適合責任）   |
| TC-007  | 民法第601条（賃貸借）    | 民法第155条＋第160条（賃貸借・敷金）  |
| TC-008  | 民法第1042条（遺留分）   | 民法第258条＋第259条（遺留分）        |
| TC-009  | 憲法第21条（表現の自由） | 憲法第17条（表現の自由）              |
| TC-011  | 憲法第81条（違憲審査）   | 憲法第71条（違憲審査権）              |
| TC-012  | 刑法第60条・61条（共犯） | 刑法第31〜33条（共同正犯・教唆・幇助）|

（GEN-062=憲法第17条、GEN-065=憲法第71条、GEN-007=刑法第151条と整合確認済み）

Usage:
    .venv/bin/python scripts/fix_test_cases.py           # dry-run（差分表示のみ）
    .venv/bin/python scripts/fix_test_cases.py --apply   # 書き込み
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CASES_DIR = PROJECT_ROOT / "test_cases"

# ケースID -> 新しい expected_statutes（全置換）
STATUTE_FIXES: dict[str, list[str]] = {
    "TC-002": ["刑法第151条"],
    "TC-003": ["刑法第198条"],
    "TC-004": ["刑法第22条"],
    "TC-005": ["民法第186条"],
    "TC-006": ["民法第147条"],
    "TC-007": ["民法第155条", "民法第160条"],
    "TC-008": ["民法第258条", "民法第259条"],
    "TC-009": ["憲法第17条"],
    "TC-011": ["憲法第71条"],
    "TC-012": ["刑法第31条", "刑法第32条", "刑法第33条"],
}

TARGET_FILES = ["all_cases.json", "default_cases.json", "generated_cases.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description="テストケース期待条文の修正")
    parser.add_argument("--apply", action="store_true", help="実際に書き込む")
    args = parser.parse_args()

    total_changes = 0
    for filename in TARGET_FILES:
        path = TEST_CASES_DIR / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        cases = data.get("test_cases", [])
        changed = []
        for tc in cases:
            fix = STATUTE_FIXES.get(tc.get("id"))
            if fix is None:
                continue
            old = tc.get("expected_statutes", [])
            if old == fix:
                continue
            changed.append((tc["id"], old, fix))
            if args.apply:
                tc["expected_statutes"] = fix

        if changed:
            print(f"--- {filename}: {len(changed)} 件 ---")
            for cid, old, new in changed:
                print(f"  {cid}: {old} -> {new}")
            total_changes += len(changed)
            if args.apply:
                path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"  => {filename} を更新しました")

    if total_changes == 0:
        print("変更対象はありません（適用済み）")
    elif not args.apply:
        print(f"\ndry-run: 合計 {total_changes} 件。--apply で書き込みます。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
