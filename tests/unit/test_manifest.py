"""Manifest model, validation, hashing, and plan-merge tests.

These cover the rules that protect the corpus: identity, enum validity, hash integrity,
reference validity, and the promise that regenerating the manifest never loses fetch state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reglens.config import get_settings
from reglens.ingestion.manifest import (
    MANIFEST_COLUMNS,
    DocumentRecord,
    load_manifest,
    plan_summary,
    raw_relative_path,
    records_from_plan,
    sha256_bytes,
    sha256_file,
    validate_records,
    verify_raw_store,
    write_manifest,
    write_manifest_schema,
)
from tests.conftest import make_record


def test_manifest_columns_are_stable() -> None:
    """The CSV header is a contract with the hand-edited plan and with git diffs."""
    assert MANIFEST_COLUMNS[0] == "doc_id"
    assert "file_hash" in MANIFEST_COLUMNS
    assert len(set(MANIFEST_COLUMNS)) == len(MANIFEST_COLUMNS)


def test_valid_rows_pass(records: list[DocumentRecord]) -> None:
    report = validate_records(records)
    assert report.ok, [issue.as_dict() for issue in report.errors]
    assert report.warnings == []


def test_duplicate_doc_id_is_an_error() -> None:
    report = validate_records([make_record(), make_record()])
    assert not report.ok
    assert any("duplicate doc_id" in issue.message for issue in report.errors)


def test_bad_doc_id_and_enum_are_errors() -> None:
    report = validate_records([make_record(doc_id="Bad ID", doc_type="memo")])
    assert not report.ok
    columns = {issue.column for issue in report.errors}
    assert {"doc_id", "doc_type"} <= columns


def test_planned_row_does_not_require_hash_or_date() -> None:
    """A planned row is a promise, not evidence: it must not need bytes or dates yet."""
    report = validate_records([make_record(status="planned")])
    assert report.ok


def test_excluded_row_does_not_require_hash() -> None:
    """Entry-point/listing rows are deliberately excluded and have nothing to hash."""
    report = validate_records([make_record(status="excluded")])
    assert report.ok, [issue.as_dict() for issue in report.errors]


def test_fetched_row_requires_hash_and_local_path() -> None:
    report = validate_records([make_record(status="fetched", issue_date="2020-09-04")])
    assert not report.ok
    assert {issue.column for issue in report.errors} == {"file_hash", "local_path"}


def test_fetched_regulatory_row_requires_issue_date() -> None:
    record = make_record(
        status="fetched",
        file_hash="a" * 64,
        local_path="rbi/rbi_md_test/rbi_md_test__aaaaaaaaaaaa.pdf",
        issue_date="",
    )
    report = validate_records([record])
    assert any(issue.column == "issue_date" for issue in report.errors)


def test_unapproved_host_is_a_warning_not_an_error() -> None:
    """Host allow-listing informs the operator; the hard gate is the fetcher."""
    report = validate_records([make_record(url="https://example.com/circular.pdf")])
    assert report.ok
    assert any(issue.column == "url" for issue in report.warnings)


def test_fiscal_period_and_date_formats() -> None:
    report = validate_records(
        [make_record(doc_type="transcript", fiscal_period="2025Q1", issue_date="04-09-2020")]
    )
    columns = {issue.column for issue in report.errors}
    assert {"fiscal_period", "issue_date"} <= columns


def test_effective_before_issue_is_a_warning() -> None:
    report = validate_records([make_record(issue_date="2025-06-01", effective_date="2025-01-01")])
    assert report.ok
    assert any(issue.column == "effective_date" for issue in report.warnings)


def test_links_must_reference_known_documents() -> None:
    report = validate_records([make_record(supersedes="does_not_exist")])
    assert not report.ok
    assert any("unknown doc_id" in issue.message for issue in report.errors)


def test_self_reference_is_rejected() -> None:
    report = validate_records([make_record(amends="rbi_md_test")])
    assert not report.ok


def test_multi_value_fields_split_on_semicolon() -> None:
    record = make_record(supersedes="a_doc;b_doc", amends="c_doc")
    assert record.multi_supersedes == ["a_doc", "b_doc"]
    assert record.multi_amends == ["c_doc"]


def test_hash_mismatch_on_disk_is_detected(records: list[DocumentRecord]) -> None:
    """Immutability guarantee: a changed raw file must fail validation."""
    raw = get_settings().raw_dir / records[0].local_path
    raw.write_bytes(b"tampered")
    report = validate_records(records)
    assert any("does not match the recorded hash" in issue.message for issue in report.errors)
    assert not verify_raw_store(records).ok


def test_missing_raw_file_is_detected(records: list[DocumentRecord]) -> None:
    (get_settings().raw_dir / records[0].local_path).unlink()
    report = validate_records(records)
    assert any("file missing on disk" in issue.message for issue in report.errors)


def test_sha256_helpers_agree(tmp_path: Path) -> None:
    payload = b"reglens"
    path = tmp_path / "x.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)


def test_roundtrip_through_csv(records: list[DocumentRecord], manifest_path: Path) -> None:
    write_manifest(records, manifest_path)
    loaded = load_manifest(manifest_path)
    assert [r.doc_id for r in loaded] == [r.doc_id for r in records]
    assert loaded[0].file_hash == records[0].file_hash


def test_raw_path_is_content_addressed() -> None:
    record = make_record()
    path = raw_relative_path(record, "abcdef1234567890" + "0" * 48)
    assert path == "rbi/rbi_md_test/rbi_md_test__abcdef123456.pdf"


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("doc_id,source\nx,rbi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest(path)


def test_unknown_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    header = ",".join([*MANIFEST_COLUMNS, "surprise"])
    path.write_text(f"{header}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown columns"):
        load_manifest(path)


def test_plan_merge_preserves_fetch_state() -> None:
    """Editing the plan and regenerating must never discard a hash or a local path."""
    fetched = make_record(status="fetched", file_hash="b" * 64, local_path="rbi/x/x__bbbb.pdf")
    plan = {
        "documents": [
            {
                "doc_id": "rbi_md_test",
                "source": "rbi",
                "issuer": "RBI",
                "doc_type": "master_direction",
                "title": "Master Direction - Test (retitled)",
                "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/164MD.PDF",
                "discovery": "direct",
            }
        ]
    }
    merged = records_from_plan(plan, [fetched])
    assert len(merged) == 1
    assert merged[0].status == "fetched"
    assert merged[0].file_hash == "b" * 64
    assert merged[0].local_path == "rbi/x/x__bbbb.pdf"
    assert merged[0].title.endswith("(retitled)")


def test_plan_merge_marks_removed_documents_excluded() -> None:
    """A document dropped from the plan stays in the manifest, marked excluded."""
    existing = make_record(status="fetched", file_hash="c" * 64, local_path="rbi/y/y__cccc.pdf")
    merged = records_from_plan({"documents": []}, [existing])
    assert len(merged) == 1
    assert merged[0].status == "excluded"
    assert "corpus_plan.yaml" in merged[0].notes


def test_plan_merge_defaults_status_and_discovery() -> None:
    """Regression: filling absent plan keys with '' wiped the model defaults."""
    merged = records_from_plan(
        {
            "documents": [
                {
                    "doc_id": "sbi_ar_fy2025",
                    "source": "bank_ir",
                    "issuer": "SBI",
                    "doc_type": "annual_report",
                    "title": "SBI Annual Report FY2024-25",
                    "url": "https://sbi.co.in/web/investor-relations",
                }
            ]
        }
    )
    assert merged[0].status == "planned"
    assert merged[0].discovery == "direct"


def test_plan_summary_counts() -> None:
    summary = plan_summary([make_record(), make_record(doc_id="x_2", source="sebi")])
    assert summary["total"] == 2
    assert summary["by_source"] == {"rbi": 1, "sebi": 1}
    assert summary["by_status"] == {"planned": 2}


def test_json_schema_matches_the_columns(tmp_path: Path) -> None:
    path = write_manifest_schema(tmp_path / "schema.json")
    import json

    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["x-manifest-columns"] == list(MANIFEST_COLUMNS)
    assert "source" in schema["x-enums"]
    assert schema["x-enums"]["status"] == ["planned", "fetched", "parsed", "failed", "excluded"]
