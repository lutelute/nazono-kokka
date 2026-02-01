"""Tests for precedent schema validation.

Validates that all precedent JSON files in the precedents/ directory
conform to the expected schema, have valid field types and formats,
and maintain referential integrity (unique case IDs, valid dates,
proper cross-references to legal framework statutes).
"""

import json
import re
from pathlib import Path

import pytest

from rag_system.config import (
    PRECEDENTS_CIVIL_DIR,
    PRECEDENTS_CONSTITUTIONAL_DIR,
    PRECEDENTS_CRIMINAL_DIR,
    PRECEDENTS_DIR,
    PRECEDENTS_METADATA_PATH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

OPTIONAL_LIST_FIELDS = ["charges", "sentence", "legal_principles"]

CASE_ID_PATTERNS: dict[str, re.Pattern] = {
    "criminal": re.compile(r"^CRIM-\d{4}-\d{4}$"),
    "civil": re.compile(r"^CIVIL-\d{4}-\d{4}$"),
    "constitutional": re.compile(r"^CONST-\d{4}-\d{4}$"),
}

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

CATEGORY_DIRS: dict[str, Path] = {
    "criminal": PRECEDENTS_CRIMINAL_DIR,
    "civil": PRECEDENTS_CIVIL_DIR,
    "constitutional": PRECEDENTS_CONSTITUTIONAL_DIR,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_all_precedent_files() -> list[tuple[str, Path]]:
    """Collect all precedent JSON files as (category, filepath) tuples."""
    results: list[tuple[str, Path]] = []
    for category, cat_dir in CATEGORY_DIRS.items():
        if cat_dir.exists():
            for filepath in sorted(cat_dir.glob("*.json")):
                results.append((category, filepath))
    return results


def _load_precedent(filepath: Path) -> dict:
    """Load and return a precedent JSON file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# Collect all files once at module level for parametrized tests
ALL_PRECEDENT_FILES = _collect_all_precedent_files()
ALL_PRECEDENT_IDS = [f"{cat}/{fp.name}" for cat, fp in ALL_PRECEDENT_FILES]


def _sample_precedent_files(n: int = 50) -> list[tuple[str, Path]]:
    """Return a deterministic sample of precedent files for heavier tests.

    Uses evenly spaced selection to cover the full range of files.
    """
    total = len(ALL_PRECEDENT_FILES)
    if total <= n:
        return list(ALL_PRECEDENT_FILES)
    step = total / n
    return [ALL_PRECEDENT_FILES[int(i * step)] for i in range(n)]


SAMPLED_FILES = _sample_precedent_files()
SAMPLED_IDS = [f"{cat}/{fp.name}" for cat, fp in SAMPLED_FILES]


# ---------------------------------------------------------------------------
# Tests: Precedent Directory Structure
# ---------------------------------------------------------------------------


class TestPrecedentDirectoryStructure:
    """Verify precedent directory layout and file counts."""

    def test_precedents_dir_exists(self):
        assert PRECEDENTS_DIR.exists(), f"Precedents directory missing: {PRECEDENTS_DIR}"

    def test_criminal_dir_exists(self):
        assert PRECEDENTS_CRIMINAL_DIR.exists()

    def test_civil_dir_exists(self):
        assert PRECEDENTS_CIVIL_DIR.exists()

    def test_constitutional_dir_exists(self):
        assert PRECEDENTS_CONSTITUTIONAL_DIR.exists()

    def test_criminal_has_precedents(self):
        files = list(PRECEDENTS_CRIMINAL_DIR.glob("*.json"))
        assert len(files) > 0, "No criminal precedent files found"

    def test_civil_has_precedents(self):
        files = list(PRECEDENTS_CIVIL_DIR.glob("*.json"))
        assert len(files) > 0, "No civil precedent files found"

    def test_constitutional_has_precedents(self):
        files = list(PRECEDENTS_CONSTITUTIONAL_DIR.glob("*.json"))
        assert len(files) > 0, "No constitutional precedent files found"

    def test_total_precedent_count_exceeds_1000(self):
        total = sum(
            len(list(cat_dir.glob("*.json")))
            for cat_dir in CATEGORY_DIRS.values()
            if cat_dir.exists()
        )
        assert total >= 1000, (
            f"Expected at least 1000 precedents, found {total}"
        )


# ---------------------------------------------------------------------------
# Tests: JSON Loadability (sampled)
# ---------------------------------------------------------------------------


class TestPrecedentJsonLoadable:
    """Verify all precedent files are valid JSON."""

    @pytest.mark.parametrize(
        "category,filepath",
        SAMPLED_FILES,
        ids=SAMPLED_IDS,
    )
    def test_json_loadable(self, category: str, filepath: Path):
        data = _load_precedent(filepath)
        assert isinstance(data, dict), f"{filepath.name}: JSON root must be a dict"


# ---------------------------------------------------------------------------
# Tests: Required Fields (sampled)
# ---------------------------------------------------------------------------


class TestPrecedentRequiredFields:
    """Verify required fields are present and non-empty in each precedent."""

    @pytest.mark.parametrize(
        "category,filepath",
        SAMPLED_FILES,
        ids=SAMPLED_IDS,
    )
    def test_required_fields_present(self, category: str, filepath: Path):
        data = _load_precedent(filepath)
        for field in REQUIRED_FIELDS:
            assert field in data, (
                f"{filepath.name}: missing required field '{field}'"
            )

    @pytest.mark.parametrize(
        "category,filepath",
        SAMPLED_FILES,
        ids=SAMPLED_IDS,
    )
    def test_required_string_fields_non_empty(self, category: str, filepath: Path):
        data = _load_precedent(filepath)
        string_fields = ["case_id", "case_type", "title", "date", "verdict", "summary", "reasoning"]
        for field in string_fields:
            value = data.get(field, "")
            assert isinstance(value, str) and value.strip(), (
                f"{filepath.name}: field '{field}' must be a non-empty string, got {value!r}"
            )


# ---------------------------------------------------------------------------
# Tests: Case ID Format (sampled)
# ---------------------------------------------------------------------------


class TestPrecedentCaseIdFormat:
    """Verify case_id matches the expected pattern for its category."""

    @pytest.mark.parametrize(
        "category,filepath",
        SAMPLED_FILES,
        ids=SAMPLED_IDS,
    )
    def test_case_id_matches_category_pattern(self, category: str, filepath: Path):
        data = _load_precedent(filepath)
        case_id = data.get("case_id", "")
        pattern = CASE_ID_PATTERNS[category]
        assert pattern.match(case_id), (
            f"{filepath.name}: case_id '{case_id}' does not match "
            f"expected pattern for {category}"
        )

    @pytest.mark.parametrize(
        "category,filepath",
        SAMPLED_FILES,
        ids=SAMPLED_IDS,
    )
    def test_case_id_matches_filename(self, category: str, filepath: Path):
        data = _load_precedent(filepath)
        case_id = data.get("case_id", "")
        expected_filename = f"{case_id}.json"
        assert filepath.name == expected_filename, (
            f"Filename '{filepath.name}' does not match case_id '{case_id}'"
        )


# ---------------------------------------------------------------------------
# Tests: Date Format (sampled)
# ---------------------------------------------------------------------------


class TestPrecedentDateFormat:
    """Verify date fields conform to YYYY-MM-DD format with valid ranges."""

    @pytest.mark.parametrize(
        "category,filepath",
        SAMPLED_FILES,
        ids=SAMPLED_IDS,
    )
    def test_date_format(self, category: str, filepath: Path):
        data = _load_precedent(filepath)
        date_str = data.get("date", "")
        assert DATE_PATTERN.match(date_str), (
            f"{filepath.name}: invalid date format '{date_str}' (expected YYYY-MM-DD)"
        )

    @pytest.mark.parametrize(
        "category,filepath",
        SAMPLED_FILES,
        ids=SAMPLED_IDS,
    )
    def test_date_valid_ranges(self, category: str, filepath: Path):
        data = _load_precedent(filepath)
        date_str = data.get("date", "")
        year, month, day = map(int, date_str.split("-"))
        assert 2000 <= year <= 2030, f"{filepath.name}: year {year} out of expected range"
        assert 1 <= month <= 12, f"{filepath.name}: invalid month {month}"
        assert 1 <= day <= 31, f"{filepath.name}: invalid day {day}"


# ---------------------------------------------------------------------------
# Tests: Referenced Statutes (sampled)
# ---------------------------------------------------------------------------


class TestPrecedentReferencedStatutes:
    """Verify referenced_statutes field is a non-empty list of strings."""

    @pytest.mark.parametrize(
        "category,filepath",
        SAMPLED_FILES,
        ids=SAMPLED_IDS,
    )
    def test_referenced_statutes_is_list(self, category: str, filepath: Path):
        data = _load_precedent(filepath)
        statutes = data.get("referenced_statutes")
        assert isinstance(statutes, list), (
            f"{filepath.name}: referenced_statutes must be a list"
        )

    @pytest.mark.parametrize(
        "category,filepath",
        SAMPLED_FILES,
        ids=SAMPLED_IDS,
    )
    def test_referenced_statutes_non_empty(self, category: str, filepath: Path):
        data = _load_precedent(filepath)
        statutes = data.get("referenced_statutes", [])
        assert len(statutes) > 0, (
            f"{filepath.name}: referenced_statutes must not be empty"
        )

    @pytest.mark.parametrize(
        "category,filepath",
        SAMPLED_FILES,
        ids=SAMPLED_IDS,
    )
    def test_referenced_statutes_contains_strings(self, category: str, filepath: Path):
        data = _load_precedent(filepath)
        statutes = data.get("referenced_statutes", [])
        for idx, statute in enumerate(statutes):
            assert isinstance(statute, str) and statute.strip(), (
                f"{filepath.name}: referenced_statutes[{idx}] must be a non-empty string"
            )


# ---------------------------------------------------------------------------
# Tests: Optional List Fields (sampled)
# ---------------------------------------------------------------------------


class TestPrecedentOptionalListFields:
    """Verify optional list fields have correct types when present."""

    @pytest.mark.parametrize(
        "category,filepath",
        SAMPLED_FILES,
        ids=SAMPLED_IDS,
    )
    def test_charges_is_list_if_present(self, category: str, filepath: Path):
        data = _load_precedent(filepath)
        if "charges" in data:
            assert isinstance(data["charges"], list), (
                f"{filepath.name}: 'charges' must be a list"
            )

    @pytest.mark.parametrize(
        "category,filepath",
        SAMPLED_FILES,
        ids=SAMPLED_IDS,
    )
    def test_legal_principles_is_list_if_present(self, category: str, filepath: Path):
        data = _load_precedent(filepath)
        if "legal_principles" in data:
            assert isinstance(data["legal_principles"], list), (
                f"{filepath.name}: 'legal_principles' must be a list"
            )


# ---------------------------------------------------------------------------
# Tests: Cross-file Integrity (run once)
# ---------------------------------------------------------------------------


class TestPrecedentCrossFileIntegrity:
    """Verify cross-file constraints like unique case IDs."""

    def test_all_case_ids_unique(self):
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for category, filepath in ALL_PRECEDENT_FILES:
            data = _load_precedent(filepath)
            case_id = data.get("case_id", "")
            if case_id in seen:
                duplicates.append(
                    f"Duplicate '{case_id}' in '{filepath.name}' and '{seen[case_id]}'"
                )
            else:
                seen[case_id] = filepath.name
        assert len(duplicates) == 0, (
            f"Found {len(duplicates)} duplicate case_id(s):\n" + "\n".join(duplicates[:10])
        )

    def test_case_ids_match_directory_prefix(self):
        """Criminal files should have CRIM- prefix, civil CIVIL-, constitutional CONST-."""
        prefix_map = {
            "criminal": "CRIM-",
            "civil": "CIVIL-",
            "constitutional": "CONST-",
        }
        mismatches: list[str] = []
        for category, filepath in ALL_PRECEDENT_FILES:
            data = _load_precedent(filepath)
            case_id = data.get("case_id", "")
            expected_prefix = prefix_map[category]
            if not case_id.startswith(expected_prefix):
                mismatches.append(
                    f"{filepath.name}: case_id '{case_id}' expected prefix '{expected_prefix}'"
                )
        assert len(mismatches) == 0, (
            f"Found {len(mismatches)} prefix mismatch(es):\n" + "\n".join(mismatches[:10])
        )


# ---------------------------------------------------------------------------
# Tests: Metadata Index
# ---------------------------------------------------------------------------


class TestPrecedentMetadata:
    """Verify metadata.json integrity."""

    def test_metadata_file_exists(self):
        assert PRECEDENTS_METADATA_PATH.exists(), (
            f"metadata.json not found at {PRECEDENTS_METADATA_PATH}"
        )

    def test_metadata_is_valid_json(self):
        with open(PRECEDENTS_METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
        assert isinstance(metadata, dict)

    def test_metadata_has_total_cases(self):
        with open(PRECEDENTS_METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
        assert "total_cases" in metadata
        assert isinstance(metadata["total_cases"], int)
        assert metadata["total_cases"] > 0

    def test_metadata_total_matches_file_count(self):
        with open(PRECEDENTS_METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
        total_files = len(ALL_PRECEDENT_FILES)
        assert metadata["total_cases"] == total_files, (
            f"metadata.json reports {metadata['total_cases']} cases "
            f"but found {total_files} files"
        )

    def test_metadata_has_statistics(self):
        with open(PRECEDENTS_METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
        assert "statistics" in metadata
        stats = metadata["statistics"]
        assert "by_category" in stats

    def test_metadata_category_counts_match(self):
        with open(PRECEDENTS_METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
        by_category = metadata.get("statistics", {}).get("by_category", {})

        for category, cat_dir in CATEGORY_DIRS.items():
            if cat_dir.exists():
                actual_count = len(list(cat_dir.glob("*.json")))
                reported_count = by_category.get(category, 0)
                assert reported_count == actual_count, (
                    f"metadata.json reports {reported_count} {category} cases "
                    f"but found {actual_count} files"
                )
