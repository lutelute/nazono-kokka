"""Data validation script for precedent integrity in 謎の国家.

Validates all precedent JSON files for schema compliance, field integrity,
case ID uniqueness, date formats, and cross-references to legal framework
statutes.

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --category criminal
    python scripts/validate_data.py --strict
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root so we can import rag_system.config
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from rag_system.config import (
    LEGAL_FRAMEWORK_DIR,
    PRECEDENTS_CIVIL_DIR,
    PRECEDENTS_CONSTITUTIONAL_DIR,
    PRECEDENTS_CRIMINAL_DIR,
    PRECEDENTS_DIR,
    PRECEDENTS_METADATA_PATH,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation Constants
# ---------------------------------------------------------------------------

ALLOWED_CASE_TYPES = {"criminal", "civil", "constitutional"}

CASE_ID_PATTERNS: dict[str, re.Pattern] = {
    "criminal": re.compile(r"^CRIM-\d{4}-\d{4}$"),
    "civil": re.compile(r"^CIVIL-\d{4}-\d{4}$"),
    "constitutional": re.compile(r"^CONST-\d{4}-\d{4}$"),
}

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

REQUIRED_FIELDS = [
    "case_id",
    "case_type",
    "title",
    "date",
    "verdict",
    "summary",
    "reasoning",
    "referenced_statutes",
]

OPTIONAL_FIELDS = [
    "charges",
    "sentence",
    "legal_principles",
]

CATEGORY_DIRS: dict[str, Path] = {
    "criminal": PRECEDENTS_CRIMINAL_DIR,
    "civil": PRECEDENTS_CIVIL_DIR,
    "constitutional": PRECEDENTS_CONSTITUTIONAL_DIR,
}

# ---------------------------------------------------------------------------
# Validation Result
# ---------------------------------------------------------------------------


class ValidationResult:
    """Accumulates validation pass/fail results with error details."""

    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_pass(self) -> None:
        self.passed += 1

    def add_failure(self, message: str) -> None:
        self.failed += 1
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def total(self) -> int:
        return self.passed + self.failed

    @property
    def is_success(self) -> bool:
        return self.failed == 0


# ---------------------------------------------------------------------------
# Individual Validators
# ---------------------------------------------------------------------------


def validate_json_loadable(filepath: Path, result: ValidationResult) -> dict | None:
    """Attempt to load a JSON file and return its content.

    Returns:
        Parsed dict on success, None on failure.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            result.add_failure(f"{filepath.name}: JSON root is not an object")
            return None
        return data
    except json.JSONDecodeError as exc:
        result.add_failure(f"{filepath.name}: invalid JSON - {exc}")
        return None
    except OSError as exc:
        result.add_failure(f"{filepath.name}: cannot read file - {exc}")
        return None


def validate_required_fields(
    data: dict, filepath: Path, result: ValidationResult
) -> bool:
    """Check that all required fields are present and non-empty."""
    all_present = True
    for field in REQUIRED_FIELDS:
        if field not in data:
            result.add_failure(f"{filepath.name}: missing required field '{field}'")
            all_present = False
        elif data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
            result.add_failure(f"{filepath.name}: field '{field}' is empty")
            all_present = False
    return all_present


def validate_case_id_format(
    data: dict, filepath: Path, result: ValidationResult,
    category: str = "",
) -> bool:
    """Validate case_id matches the expected pattern for its category.

    Args:
        data: Parsed precedent JSON data.
        filepath: Path to the precedent file.
        result: ValidationResult to record findings.
        category: The directory-level category name (e.g., 'criminal',
            'civil', 'constitutional') used to look up the expected ID
            pattern.  This is distinct from the JSON ``case_type`` field
            which contains Japanese subcategory names.
    """
    case_id = data.get("case_id", "")

    if category not in CASE_ID_PATTERNS:
        result.add_failure(
            f"{filepath.name}: cannot validate case_id - unknown category '{category}'"
        )
        return False

    pattern = CASE_ID_PATTERNS[category]
    if not pattern.match(case_id):
        result.add_failure(
            f"{filepath.name}: case_id '{case_id}' does not match "
            f"expected pattern for category '{category}'"
        )
        return False
    return True


def validate_case_type(
    data: dict, filepath: Path, result: ValidationResult
) -> bool:
    """Validate case_type is a non-empty string.

    The case_type field contains Japanese subcategory names (e.g., '窃盗',
    '売買契約', '表現の自由'), so we only check that it is present and
    non-empty rather than matching against a fixed set.
    """
    case_type = data.get("case_type", "")
    if not isinstance(case_type, str) or not case_type.strip():
        result.add_failure(
            f"{filepath.name}: case_type must be a non-empty string, "
            f"got '{case_type}'"
        )
        return False
    return True


def validate_date_format(
    data: dict, filepath: Path, result: ValidationResult
) -> bool:
    """Validate date field matches YYYY-MM-DD format."""
    date_str = data.get("date", "")
    if not DATE_PATTERN.match(str(date_str)):
        result.add_failure(
            f"{filepath.name}: invalid date format '{date_str}' "
            f"(expected YYYY-MM-DD)"
        )
        return False

    # Basic range validation
    try:
        year, month, day = map(int, str(date_str).split("-"))
        if not (1 <= month <= 12):
            result.add_failure(
                f"{filepath.name}: invalid month {month} in date '{date_str}'"
            )
            return False
        if not (1 <= day <= 31):
            result.add_failure(
                f"{filepath.name}: invalid day {day} in date '{date_str}'"
            )
            return False
    except ValueError:
        result.add_failure(
            f"{filepath.name}: cannot parse date '{date_str}'"
        )
        return False

    return True


def validate_referenced_statutes(
    data: dict, filepath: Path, result: ValidationResult
) -> bool:
    """Validate that referenced_statutes is a non-empty list of strings."""
    statutes = data.get("referenced_statutes")
    if statutes is None:
        result.add_failure(
            f"{filepath.name}: referenced_statutes is missing"
        )
        return False
    if not isinstance(statutes, list):
        result.add_failure(
            f"{filepath.name}: referenced_statutes must be a list, "
            f"got {type(statutes).__name__}"
        )
        return False
    if len(statutes) == 0:
        result.add_failure(
            f"{filepath.name}: referenced_statutes is empty"
        )
        return False
    for idx, statute in enumerate(statutes):
        if not isinstance(statute, str) or not statute.strip():
            result.add_failure(
                f"{filepath.name}: referenced_statutes[{idx}] is not a valid string"
            )
            return False
    return True


def validate_list_fields(
    data: dict, filepath: Path, result: ValidationResult
) -> bool:
    """Validate optional list fields (charges, legal_principles) if present."""
    valid = True
    for field in ("charges", "legal_principles"):
        if field in data:
            value = data[field]
            if not isinstance(value, list):
                result.add_failure(
                    f"{filepath.name}: '{field}' must be a list, "
                    f"got {type(value).__name__}"
                )
                valid = False
    return valid


# ---------------------------------------------------------------------------
# File-level Validation
# ---------------------------------------------------------------------------


def validate_precedent_file(
    filepath: Path, result: ValidationResult, category: str = ""
) -> dict | None:
    """Run all validations on a single precedent JSON file.

    Args:
        filepath: Path to the JSON precedent file.
        result: ValidationResult to record findings.
        category: The directory-level category name (e.g., 'criminal',
            'civil', 'constitutional') passed through to
            ``validate_case_id_format``.

    Returns:
        Parsed data dict on success, None on failure.
    """
    data = validate_json_loadable(filepath, result)
    if data is None:
        return None

    checks = [
        validate_required_fields(data, filepath, result),
        validate_case_type(data, filepath, result),
        validate_case_id_format(data, filepath, result, category=category),
        validate_date_format(data, filepath, result),
        validate_referenced_statutes(data, filepath, result),
        validate_list_fields(data, filepath, result),
    ]

    if all(checks):
        result.add_pass()
        return data
    return data


# ---------------------------------------------------------------------------
# Cross-file Validations
# ---------------------------------------------------------------------------


def validate_case_id_uniqueness(
    all_case_ids: list[tuple[str, str]],
    result: ValidationResult,
) -> None:
    """Check that all case_ids are unique across the entire precedent database.

    Args:
        all_case_ids: List of (case_id, filename) tuples.
        result: ValidationResult to record findings.
    """
    seen: dict[str, str] = {}
    for case_id, filename in all_case_ids:
        if case_id in seen:
            result.add_failure(
                f"Duplicate case_id '{case_id}' found in "
                f"'{filename}' and '{seen[case_id]}'"
            )
        else:
            seen[case_id] = filename


def validate_metadata_index(
    all_case_ids: set[str],
    result: ValidationResult,
) -> None:
    """Validate metadata.json references all discovered precedents.

    Args:
        all_case_ids: Set of all case_ids found in individual files.
        result: ValidationResult to record findings.
    """
    if not PRECEDENTS_METADATA_PATH.exists():
        result.add_warning(
            f"metadata.json not found at {PRECEDENTS_METADATA_PATH} - "
            f"skipping metadata validation"
        )
        return

    try:
        with open(PRECEDENTS_METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        result.add_failure(f"metadata.json: cannot load - {exc}")
        return

    cases = metadata.get("cases", [])
    if not isinstance(cases, list):
        result.add_failure("metadata.json: 'cases' field must be a list")
        return

    indexed_ids = set()
    for entry in cases:
        if isinstance(entry, dict) and "case_id" in entry:
            indexed_ids.add(entry["case_id"])

    # Check for precedents missing from index
    missing_from_index = all_case_ids - indexed_ids
    if missing_from_index:
        result.add_warning(
            f"metadata.json: {len(missing_from_index)} precedent(s) not indexed "
            f"(e.g., {list(missing_from_index)[:3]})"
        )

    # Check for phantom entries in index
    phantom_entries = indexed_ids - all_case_ids
    if phantom_entries:
        result.add_warning(
            f"metadata.json: {len(phantom_entries)} indexed case_id(s) have no "
            f"corresponding file (e.g., {list(phantom_entries)[:3]})"
        )

    logger.info(
        "metadata.json: %d entries indexed, %d precedent files found",
        len(indexed_ids),
        len(all_case_ids),
    )


def validate_legal_framework_exists(result: ValidationResult) -> None:
    """Check that the legal framework documents exist."""
    expected_files = [
        "constitution.md",
        "criminal_code.md",
        "civil_code.md",
        "cultural_regulations.md",
        "ethical_guidelines.md",
        "administrative_code.md",
    ]
    for filename in expected_files:
        filepath = LEGAL_FRAMEWORK_DIR / filename
        if not filepath.exists():
            result.add_warning(
                f"Legal framework file missing: {filepath}"
            )
        elif filepath.stat().st_size == 0:
            result.add_warning(
                f"Legal framework file is empty: {filepath}"
            )


# ---------------------------------------------------------------------------
# Category Scanning
# ---------------------------------------------------------------------------


def collect_precedent_files(categories: list[str] | None = None) -> dict[str, list[Path]]:
    """Collect all precedent JSON files grouped by category.

    Args:
        categories: Optional list of categories to scan. Defaults to all.

    Returns:
        Dict mapping category name to list of JSON file paths.
    """
    if categories is None:
        categories = list(CATEGORY_DIRS.keys())

    files_by_category: dict[str, list[Path]] = {}
    for category in categories:
        cat_dir = CATEGORY_DIRS.get(category)
        if cat_dir is None:
            logger.warning("Unknown category: %s", category)
            continue
        if not cat_dir.exists():
            logger.warning("Category directory does not exist: %s", cat_dir)
            files_by_category[category] = []
            continue

        json_files = sorted(cat_dir.glob("*.json"))
        files_by_category[category] = json_files

    return files_by_category


# ---------------------------------------------------------------------------
# Main Validation Pipeline
# ---------------------------------------------------------------------------


def run_validation(
    categories: list[str] | None = None,
    strict: bool = False,
) -> ValidationResult:
    """Run the full validation pipeline.

    Args:
        categories: Categories to validate (None for all).
        strict: If True, treat warnings as failures.

    Returns:
        Aggregated ValidationResult.
    """
    result = ValidationResult()

    # Phase 1: Check legal framework existence
    logger.info("--- 法的枠組みファイルの確認 ---")
    validate_legal_framework_exists(result)

    # Phase 2: Collect and validate individual precedent files
    logger.info("--- 判例ファイルの検証 ---")
    files_by_category = collect_precedent_files(categories)

    all_case_ids: list[tuple[str, str]] = []
    total_files = sum(len(files) for files in files_by_category.values())

    if total_files == 0:
        result.add_warning(
            "No precedent files found. Run generate_precedents.py first."
        )
        logger.warning("判例ファイルが見つかりません。")
    else:
        logger.info("検証対象ファイル数: %d", total_files)

    for category, files in files_by_category.items():
        logger.info("[%s] %d ファイルを検証中...", category, len(files))
        for filepath in files:
            data = validate_precedent_file(filepath, result, category=category)
            if data and "case_id" in data:
                all_case_ids.append((data["case_id"], filepath.name))

    # Phase 3: Cross-file validations
    if all_case_ids:
        logger.info("--- クロスファイル検証 ---")
        validate_case_id_uniqueness(all_case_ids, result)

        all_ids_set = {cid for cid, _ in all_case_ids}
        validate_metadata_index(all_ids_set, result)

    # Strict mode: promote warnings to failures
    if strict and result.warnings:
        for warning in result.warnings:
            result.add_failure(f"[strict] {warning}")
        result.warnings.clear()

    return result


def print_report(result: ValidationResult) -> None:
    """Print a human-readable validation report."""
    logger.info("=== 検証結果レポート ===")
    logger.info("合格: %d", result.passed)
    logger.info("失敗: %d", result.failed)
    logger.info("合計: %d", result.total)

    if result.warnings:
        logger.info("--- 警告 (%d) ---", len(result.warnings))
        for warning in result.warnings:
            logger.warning("  %s", warning)

    if result.errors:
        logger.info("--- エラー (%d) ---", len(result.errors))
        for error in result.errors:
            logger.error("  %s", error)

    if result.is_success:
        logger.info("結果: 全ての検証に合格しました。")
    else:
        logger.error(
            "結果: %d 件の検証に失敗しました。", result.failed
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="謎の国家の判例データの整合性を検証するスクリプト",
    )
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_DIRS.keys()),
        default=None,
        help="検証するカテゴリを指定（省略時は全カテゴリを検証）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="厳格モード: 警告もエラーとして扱う",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the data validation script."""
    args = parse_args()

    logger.info("=== 謎の国家 データ検証スクリプト ===")

    categories = [args.category] if args.category else None
    result = run_validation(categories=categories, strict=args.strict)

    print_report(result)

    if not result.is_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
