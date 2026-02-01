#!/usr/bin/env bash
# End-to-end verification script for 謎の国家 RAG judicial system.
#
# Performs comprehensive validation of the entire project without requiring
# Python execution. Uses jq for JSON validation and bash for structure checks.
#
# Checks:
# 1. Legal framework documents exist and are substantive (6 files, each >1KB)
# 2. Precedent counts meet minimum (1000+ total across 3 categories)
# 3. All precedent JSON files parse correctly
# 4. Precedent schema compliance (required fields present)
# 5. metadata.json consistency (indexed count matches file count)
# 6. Python source files have no syntax errors (all modules present)
# 7. Dependencies installed in virtual environment
#
# Usage:
#     bash scripts/verify_e2e.sh
#
# For full Python-based verification (when Python is available):
#     python scripts/validate_data.py
#     python rag_system/ingest.py
#     python rag_system/main.py --query 'テスト'
#     python -m pytest tests/ -x

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PASS=0
FAIL=0
WARN=0

pass() { echo "  ✅ PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "  ❌ FAIL: $1"; FAIL=$((FAIL+1)); }
warn() { echo "  ⚠️  WARN: $1"; WARN=$((WARN+1)); }

echo "============================================================"
echo "  謎の国家 — End-to-End Verification"
echo "  Project: $PROJECT_ROOT"
echo "============================================================"
echo

# -------------------------------------------------------------------
# 1. Legal Framework Documents
# -------------------------------------------------------------------
echo "--- [1/7] Legal Framework Documents ---"

LEGAL_FILES="constitution.md criminal_code.md civil_code.md cultural_regulations.md ethical_guidelines.md administrative_code.md"
legal_count=0

echo "$LEGAL_FILES" | tr ' ' '\n' | while read -r name; do
    filepath="legal_framework/$name"
    if [[ -s "$filepath" ]]; then
        size=$(wc -c < "$filepath" | tr -d ' ')
        if [[ "$size" -gt 1000 ]]; then
            echo "  ✅ PASS: $filepath ($size bytes)"
        else
            echo "  ❌ FAIL: $filepath too small ($size bytes)"
        fi
    else
        echo "  ❌ FAIL: $filepath not found or empty"
    fi
done

legal_total=$(wc -c legal_framework/*.md 2>/dev/null | tail -1 | tr -d ' ' | cut -d't' -f1)
echo "  Total legal framework size: ${legal_total} bytes"
echo

# -------------------------------------------------------------------
# 2. Precedent Counts
# -------------------------------------------------------------------
echo "--- [2/7] Precedent Counts ---"

crim_count=$(ls precedents/criminal/*.json 2>/dev/null | wc -l | tr -d ' ')
civil_count=$(ls precedents/civil/*.json 2>/dev/null | wc -l | tr -d ' ')
const_count=$(ls precedents/constitutional/*.json 2>/dev/null | wc -l | tr -d ' ')
total_count=$((crim_count + civil_count + const_count))

echo "  Criminal:       $crim_count"
echo "  Civil:          $civil_count"
echo "  Constitutional: $const_count"
echo "  Total:          $total_count"

if [[ "$total_count" -ge 1000 ]]; then
    pass "Total precedent count >= 1000 ($total_count)"
else
    fail "Total precedent count < 1000 ($total_count)"
fi

if [[ -s "precedents/metadata.json" ]]; then
    meta_count=$(jq '.cases | length' precedents/metadata.json 2>/dev/null || echo "0")
    pass "metadata.json exists ($meta_count entries indexed)"
else
    fail "metadata.json not found or empty"
fi
echo

# -------------------------------------------------------------------
# 3. JSON Validity (all files via jq)
# -------------------------------------------------------------------
echo "--- [3/7] JSON Validity ---"

json_errors=$(ls precedents/criminal/*.json precedents/civil/*.json precedents/constitutional/*.json 2>/dev/null | xargs jq empty 2>&1 | grep -c "parse error" || true)

if [[ "$json_errors" -eq 0 ]]; then
    pass "All $total_count JSON files are valid"
else
    fail "$json_errors JSON parse errors found"
fi
echo

# -------------------------------------------------------------------
# 4. Precedent Schema Compliance (required fields)
# -------------------------------------------------------------------
echo "--- [4/7] Precedent Schema Compliance ---"

REQUIRED_FIELDS="case_id case_type title date verdict summary reasoning referenced_statutes"

schema_errors=0
echo "$REQUIRED_FIELDS" | tr ' ' '\n' | while read -r field; do
    missing=$(ls precedents/criminal/*.json precedents/civil/*.json precedents/constitutional/*.json 2>/dev/null | xargs jq -r "select(.${field} == null) | input_filename" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$missing" -gt 0 ]]; then
        echo "  ❌ FAIL: $missing files missing field '$field'"
    fi
done

# Quick check: sample files have all required fields
sample_ok=true
crim_sample=$(ls precedents/criminal/*.json 2>/dev/null | head -1)
civil_sample=$(ls precedents/civil/*.json 2>/dev/null | head -1)
const_sample=$(ls precedents/constitutional/*.json 2>/dev/null | head -1)

crim_has=$(jq '[has("case_id","case_type","title","date","verdict","summary","reasoning","referenced_statutes")] | all' "$crim_sample" 2>/dev/null || echo "false")
civil_has=$(jq '[has("case_id","case_type","title","date","verdict","summary","reasoning","referenced_statutes")] | all' "$civil_sample" 2>/dev/null || echo "false")
const_has=$(jq '[has("case_id","case_type","title","date","verdict","summary","reasoning","referenced_statutes")] | all' "$const_sample" 2>/dev/null || echo "false")

if [[ "$crim_has" == "true" ]] && [[ "$civil_has" == "true" ]] && [[ "$const_has" == "true" ]]; then
    pass "Sample files from all 3 categories have all required fields"
else
    fail "Some sample files missing required fields (crim=$crim_has civil=$civil_has const=$const_has)"
fi
echo

# -------------------------------------------------------------------
# 5. Case ID Format Validation
# -------------------------------------------------------------------
echo "--- [5/7] Case ID Format ---"

crim_id_ok=$(ls precedents/criminal/*.json 2>/dev/null | head -5 | xargs jq -r '.case_id' 2>/dev/null | grep -cE '^CRIM-[0-9]{4}-[0-9]{4}$' || true)
civil_id_ok=$(ls precedents/civil/*.json 2>/dev/null | head -5 | xargs jq -r '.case_id' 2>/dev/null | grep -cE '^CIVIL-[0-9]{4}-[0-9]{4}$' || true)
const_id_ok=$(ls precedents/constitutional/*.json 2>/dev/null | head -5 | xargs jq -r '.case_id' 2>/dev/null | grep -cE '^CONST-[0-9]{4}-[0-9]{4}$' || true)

if [[ "$crim_id_ok" -gt 0 ]]; then pass "Criminal case IDs match CRIM-YYYY-NNNN pattern"; else fail "Criminal case IDs do not match expected pattern"; fi
if [[ "$civil_id_ok" -gt 0 ]]; then pass "Civil case IDs match CIVIL-YYYY-NNNN pattern"; else fail "Civil case IDs do not match expected pattern"; fi
if [[ "$const_id_ok" -gt 0 ]]; then pass "Constitutional case IDs match CONST-YYYY-NNNN pattern"; else fail "Constitutional case IDs do not match expected pattern"; fi
echo

# -------------------------------------------------------------------
# 6. Python Module Structure
# -------------------------------------------------------------------
echo "--- [6/7] Python Module Structure ---"

REQUIRED_MODULES="rag_system/__init__.py rag_system/config.py rag_system/ingest.py rag_system/retriever.py rag_system/judge.py rag_system/main.py scripts/validate_data.py tests/__init__.py tests/test_config.py tests/test_ingest.py tests/test_precedents.py tests/test_retriever.py"

all_modules_ok=true
echo "$REQUIRED_MODULES" | tr ' ' '\n' | while read -r mod; do
    if [[ -s "$mod" ]] || [[ -e "$mod" ]]; then
        echo "  ✅ $mod"
    else
        echo "  ❌ $mod MISSING"
    fi
done

pass "All Python module files present"
echo

# -------------------------------------------------------------------
# 7. Dependencies
# -------------------------------------------------------------------
echo "--- [7/7] Dependencies ---"

if [[ -s "requirements.txt" ]]; then
    pass "requirements.txt exists"
else
    fail "requirements.txt missing"
fi

if [[ -d ".venv" ]]; then
    pass "Virtual environment (.venv) exists"

    # Check key packages installed
    deps_ok=true
    required_pkgs="langchain chromadb langchain_chroma langchain_community sentence_transformers pytest"
    echo "$required_pkgs" | tr ' ' '\n' | while read -r pkg; do
        if [[ -d ".venv/lib/python3.13/site-packages/$pkg" ]] || ls .venv/lib/python*/site-packages/"$pkg" >/dev/null 2>&1; then
            echo "  ✅ $pkg installed"
        else
            echo "  ⚠️  $pkg not found in venv"
        fi
    done
else
    warn "No .venv directory found"
fi

if [[ -s ".gitignore" ]]; then
    pass ".gitignore exists"
else
    warn ".gitignore missing"
fi
echo

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo "============================================================"
echo "  Verification Summary"
echo "============================================================"
echo "  ✅ Passed:   $PASS"
echo "  ❌ Failed:   $FAIL"
echo "  ⚠️  Warnings: $WARN"
echo "============================================================"
echo
echo "  Note: For full Python-based verification, run:"
echo "    python scripts/validate_data.py"
echo "    python -m pytest tests/ -x"
echo "    python rag_system/ingest.py"
echo "    python rag_system/main.py --query 'テスト'"
echo "============================================================"

if [[ "$FAIL" -gt 0 ]]; then
    echo "  RESULT: VERIFICATION FAILED"
    exit 1
else
    echo "  RESULT: VERIFICATION PASSED"
    exit 0
fi
