"""テストケース管理モジュール。

検証用テストケースのCRUD操作（ロード、保存、追加、更新、削除、
インポート・エクスポート）を提供する。
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from rag_system.config import PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_CASES_DIR = PROJECT_ROOT / "test_cases"
DEFAULT_CASES_PATH = TEST_CASES_DIR / "default_cases.json"


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """検証用テストケースを表すデータクラス。"""

    __test__ = False  # Prevent pytest from collecting this as a test class

    id: str
    category: str  # "criminal", "civil", "constitutional"
    query: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_statutes: list[str] = field(default_factory=list)
    expected_case_ids: list[str] = field(default_factory=list)
    difficulty: str = "basic"  # "basic", "intermediate", "advanced"
    description: str = ""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換する。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestCase":
        """辞書からインスタンスを生成する。"""
        return cls(
            id=data.get("id", _generate_id()),
            category=data.get("category", ""),
            query=data.get("query", ""),
            expected_keywords=data.get("expected_keywords", []),
            expected_statutes=data.get("expected_statutes", []),
            expected_case_ids=data.get("expected_case_ids", []),
            difficulty=data.get("difficulty", "basic"),
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class TestCaseManager:
    """テストケースのCRUD操作を提供するマネージャクラス。"""

    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._storage_dir = storage_dir or TEST_CASES_DIR
        self._cases: dict[str, TestCase] = {}

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load_default_cases(self) -> list[TestCase]:
        """デフォルトケースファイルからテストケースを読み込む。"""
        return self.load_from_file(DEFAULT_CASES_PATH)

    def load_from_file(self, path: Path) -> list[TestCase]:
        """指定されたJSONファイルからテストケースを読み込む。"""
        path = Path(path)
        if not path.exists():
            logger.warning("テストケースファイルが見つかりません: %s", path)
            return []

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("テストケースファイルの読み込みに失敗しました: %s", exc)
            return []

        raw_cases = data.get("test_cases", [])
        cases: list[TestCase] = []
        for item in raw_cases:
            tc = TestCase.from_dict(item)
            self._cases[tc.id] = tc
            cases.append(tc)

        logger.info("%d 件のテストケースを読み込みました: %s", len(cases), path)
        return cases

    def save_to_file(self, path: Path | None = None) -> None:
        """現在のテストケースをJSONファイルに保存する。"""
        path = Path(path) if path is not None else DEFAULT_CASES_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "test_cases": [tc.to_dict() for tc in self._cases.values()],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(
                "%d 件のテストケースを保存しました: %s",
                len(self._cases),
                path,
            )
        except OSError as exc:
            logger.error("テストケースの保存に失敗しました: %s", exc)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_cases(self) -> list[TestCase]:
        """登録済みの全テストケースをリストで返す。"""
        return list(self._cases.values())

    def get(self, case_id: str) -> TestCase | None:
        """IDでテストケースを取得する。見つからない場合は ``None``。"""
        return self._cases.get(case_id)

    def add(self, test_case: TestCase) -> TestCase:
        """テストケースを追加する。"""
        if test_case.id in self._cases:
            logger.warning("テストケースID '%s' は既に存在します。上書きします。", test_case.id)
        self._cases[test_case.id] = test_case
        logger.info("テストケースを追加しました: %s", test_case.id)
        return test_case

    def update(self, case_id: str, **kwargs: Any) -> TestCase | None:
        """既存テストケースのフィールドを更新する。"""
        tc = self._cases.get(case_id)
        if tc is None:
            logger.warning("更新対象のテストケースが見つかりません: %s", case_id)
            return None

        for key, value in kwargs.items():
            if hasattr(tc, key) and key != "id":
                setattr(tc, key, value)
        logger.info("テストケースを更新しました: %s", case_id)
        return tc

    def delete(self, case_id: str) -> bool:
        """テストケースを削除する。成功時に ``True`` を返す。"""
        if case_id not in self._cases:
            logger.warning("削除対象のテストケースが見つかりません: %s", case_id)
            return False
        del self._cases[case_id]
        logger.info("テストケースを削除しました: %s", case_id)
        return True

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def import_cases(self, path: Path) -> list[TestCase]:
        """外部JSONファイルからテストケースをインポートする。

        既存のケースとマージされ、同一IDは上書きされる。
        """
        imported = self.load_from_file(path)
        logger.info("%d 件のテストケースをインポートしました", len(imported))
        return imported

    def export_cases(self, path: Path) -> None:
        """現在のテストケースを指定パスにエクスポートする。"""
        self.save_to_file(path)
        logger.info("テストケースをエクスポートしました: %s", path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_id() -> str:
    """ユニークなテストケースIDを生成する。"""
    return f"TC-{uuid.uuid4().hex[:8].upper()}"
