"""Fetcher tests.

All network access is mocked with ``httpx.MockTransport``. The properties under test are
the ones that make automated collection acceptable: robots.txt is obeyed, requests are
throttled, oversized responses are refused, re-running is free, and nothing is fetched
without an explicit terms-of-use confirmation.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from reglens.config import get_settings
from reglens.ingestion.fetch import (
    FetchPolicy,
    RobotsCache,
    _Throttle,
    download_manifest,
    extract_links,
    fetch_bytes,
    filter_document_links,
    plan_downloads,
    referer_for,
    resolve_listing,
)
from reglens.ingestion.manifest import load_manifest
from tests.conftest import make_record

PDF_BYTES = b"%PDF-1.6\n" + b"reglens" * 100


def handler_factory(robots: str = "User-agent: *\nAllow: /\n", status: int = 200) -> callable:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=robots, request=request)
        return httpx.Response(
            status,
            content=PDF_BYTES,
            headers={"content-type": "application/pdf"},
            request=request,
        )

    return handler


def make_client(policy: FetchPolicy, handler: callable) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler), headers={"User-Agent": policy.user_agent}
    )


def policy(**overrides: object) -> FetchPolicy:
    base = {"delay_seconds": 0.0, "timeout_seconds": 5.0, "max_attempts": 1}
    base.update(overrides)
    return FetchPolicy(**base)


# ------------------------------------------------------------------ robots
def test_robots_allows_when_permitted() -> None:
    p = policy()
    client = make_client(p, handler_factory())
    allowed, reason = RobotsCache(p).allowed(
        client, "https://www.sebi.gov.in/legal/circulars/x.html"
    )
    assert allowed
    assert "allowed by robots.txt" in reason


def test_robots_blocks_disallowed_paths() -> None:
    robots = "User-agent: *\nDisallow: /legal\n"
    p = policy()
    client = make_client(p, handler_factory(robots=robots))
    allowed, reason = RobotsCache(p).allowed(client, "https://www.sebi.gov.in/legal/x.html")
    assert not allowed
    assert "robots.txt" in reason


def test_robots_blocks_the_download_path() -> None:
    p = policy()
    client = make_client(p, handler_factory(robots="User-agent: *\nDisallow: /legal\n"))
    result = fetch_bytes(
        "https://www.sebi.gov.in/legal/x.pdf",
        client=client,
        policy=p,
        robots=RobotsCache(p),
        throttle=_Throttle(0.0),
    )
    assert not result.ok
    assert "robots" in (result.error or "")
    assert result.content == b""


def test_unpublished_robots_txt_allows_with_a_reason() -> None:
    """RBI's robots.txt answers 418 from its WAF; that must not be read as a denial."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(418, text="Unauthorised Access", request=request)
        return httpx.Response(200, content=PDF_BYTES, request=request)

    p = policy()
    client = make_client(p, handler)
    allowed, reason = RobotsCache(p).allowed(client, "https://www.rbi.org.in/Scripts/x.aspx")
    assert allowed
    assert "unavailable" in reason


def test_robots_cache_queries_each_host_once() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n", request=request)
        return httpx.Response(200, content=PDF_BYTES, request=request)

    p = policy()
    client = make_client(p, handler)
    cache = RobotsCache(p)
    for _ in range(3):
        cache.allowed(client, "https://www.sebi.gov.in/a.pdf")
    assert calls.count("/robots.txt") == 1


def test_robots_checking_can_be_disabled() -> None:
    p = policy(respect_robots=False)
    client = make_client(p, handler_factory(robots="User-agent: *\nDisallow: /\n"))
    allowed, reason = RobotsCache(p).allowed(client, "https://x.test/y.pdf")
    assert allowed
    assert "disabled" in reason


# ------------------------------------------------------------------ fetching
def test_fetch_returns_content_and_status() -> None:
    p = policy()
    client = make_client(p, handler_factory())
    result = fetch_bytes(
        "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/164MD.PDF",
        client=client,
        policy=p,
        robots=RobotsCache(p),
        throttle=_Throttle(0.0),
    )
    assert result.ok
    assert result.status_code == 200
    assert result.content == PDF_BYTES
    assert result.suffix == ".pdf"


def test_referer_is_sent_only_for_configured_hosts() -> None:
    """rbidocs serves an HTML interstitial without a Referer (verified 2026-10-02)."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, request=request)
        seen.append(request.headers.get("referer"))
        return httpx.Response(200, content=PDF_BYTES, request=request)

    p = policy()
    client = make_client(p, handler)
    robots = RobotsCache(p)
    fetch_bytes(
        "https://rbidocs.rbi.org.in/x.PDF",
        client=client,
        policy=p,
        robots=robots,
        throttle=_Throttle(0.0),
    )
    fetch_bytes(
        "https://www.sebi.gov.in/x.pdf",
        client=client,
        policy=p,
        robots=robots,
        throttle=_Throttle(0.0),
    )
    assert seen[0] == "https://m.rbi.org.in/"
    assert seen[1] is None
    assert referer_for("https://www.sebi.gov.in/x.pdf", p) is None


def test_oversized_response_is_refused() -> None:
    p = policy(max_bytes=10)
    client = make_client(p, handler_factory())
    result = fetch_bytes(
        "https://rbidocs.rbi.org.in/big.PDF",
        client=client,
        policy=p,
        robots=RobotsCache(p),
        throttle=_Throttle(0.0),
    )
    assert not result.ok
    assert "max_bytes" in (result.error or "")


def test_declared_oversized_content_length_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, request=request)
        return httpx.Response(
            200, headers={"content-length": "999999999"}, content=b"x", request=request
        )

    p = policy(max_bytes=1000)
    client = make_client(p, handler)
    result = fetch_bytes(
        "https://rbidocs.rbi.org.in/big.PDF",
        client=client,
        policy=p,
        robots=RobotsCache(p),
        throttle=_Throttle(0.0),
    )
    assert not result.ok


def test_http_error_is_reported_not_raised() -> None:
    p = policy()
    client = make_client(p, handler_factory(status=404))
    result = fetch_bytes(
        "https://rbidocs.rbi.org.in/missing.PDF",
        client=client,
        policy=p,
        robots=RobotsCache(p),
        throttle=_Throttle(0.0),
    )
    assert not result.ok
    assert result.status_code == 404


# ------------------------------------------------------------------ planning
def test_plan_skips_documents_whose_hash_matches(records) -> None:
    actions = plan_downloads(records, raw_dir=get_settings().raw_dir)
    by_id = {action.doc_id: action for action in actions}
    assert by_id["rbi_md_test"].kind == "skip_local"
    assert by_id["sebi_mc_test"].kind == "download"


def test_plan_redownloads_when_the_local_file_is_missing(records) -> None:
    (get_settings().raw_dir / records[0].local_path).unlink()
    actions = plan_downloads(records, raw_dir=get_settings().raw_dir)
    assert {action.doc_id: action.kind for action in actions}["rbi_md_test"] == "download"


def test_plan_marks_excluded_rows(records) -> None:
    from reglens.ingestion.manifest import DocumentRecord

    excluded = DocumentRecord.model_validate(
        {**records[1].model_dump(), "doc_id": "sebi_index", "status": "excluded"}
    )
    actions = plan_downloads([excluded], raw_dir=get_settings().raw_dir)
    assert actions[0].kind == "excluded"


def test_plan_reports_rows_without_a_url(records) -> None:
    broken = records[1].model_copy(update={"url": ""})
    actions = plan_downloads([broken], raw_dir=get_settings().raw_dir)
    assert actions[0].kind == "error"


def test_plan_limit_applies_only_to_downloads(records) -> None:
    actions = plan_downloads(records, raw_dir=get_settings().raw_dir, limit=0)
    assert {action.kind for action in actions} == {"skip_local"}


def test_plan_only_filter(records) -> None:
    actions = plan_downloads(records, raw_dir=get_settings().raw_dir, only={"sebi_mc_test"})
    assert [action.doc_id for action in actions] == ["sebi_mc_test"]


# ------------------------------------------------------------------ download runs
def test_dry_run_makes_no_requests(records) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        calls.append(str(request.url))
        return httpx.Response(500, request=request)

    report = download_manifest(
        records,
        policy=policy(),
        dry_run=True,
        client=make_client(policy(), handler),
    )
    assert calls == []
    assert report.dry_run
    assert report.counts()["download"] == 1


def test_download_requires_terms_confirmation(records) -> None:
    with pytest.raises(PermissionError, match="terms of use"):
        download_manifest(records, policy=policy(), dry_run=False, accept_terms=False)


def test_download_writes_hashed_immutable_files_and_updates_manifest(records) -> None:
    manifest_path = get_settings().manifest_path
    from reglens.ingestion.manifest import write_manifest

    write_manifest(records, manifest_path)
    client = make_client(policy(), handler_factory())
    report = download_manifest(
        records, policy=policy(), dry_run=False, accept_terms=True, client=client
    )
    assert report.downloaded == ["sebi_mc_test"]
    assert report.failures == []

    reloaded = {record.doc_id: record for record in load_manifest(manifest_path)}
    fetched = reloaded["sebi_mc_test"]
    assert fetched.status == "fetched"
    assert len(fetched.file_hash) == 64
    assert fetched.local_path.startswith("sebi/sebi_mc_test/sebi_mc_test__")
    assert (get_settings().raw_dir / fetched.local_path).read_bytes() == PDF_BYTES

    # Re-running is free: the hash matches, so the document is skipped.
    second = download_manifest(
        list(reloaded.values()),
        policy=policy(),
        dry_run=False,
        accept_terms=True,
        client=client,
    )
    assert second.downloaded == []
    assert second.counts()["skip_local"] == 2


def test_failed_fetch_is_recorded_without_losing_the_row(records) -> None:
    manifest_path = get_settings().manifest_path
    from reglens.ingestion.manifest import write_manifest

    write_manifest(records, manifest_path)
    client = make_client(policy(), handler_factory(status=503))
    report = download_manifest(
        records, policy=policy(), dry_run=False, accept_terms=True, client=client
    )
    assert report.failures
    reloaded = {record.doc_id: record for record in load_manifest(manifest_path)}
    assert reloaded["sebi_mc_test"].status == "failed"
    assert "fetch failed" in reloaded["sebi_mc_test"].notes


# ------------------------------------------------------------------ listing pages
def test_extract_links_resolves_relative_urls() -> None:
    html = '<a href="/a/b.pdf">x</a><a href="https://x.test/c.pdf">y</a><a href="/a/b.pdf">dup</a>'
    links = extract_links(html, "https://x.test/base/page.html")
    assert links == ["https://x.test/a/b.pdf", "https://x.test/c.pdf"]


def test_filter_document_links_matches_patterns_and_required_parts() -> None:
    links = [
        "https://x.test/legal/annual-report-fy2025.pdf",
        "https://x.test/legal/transcript-q4.pdf",
        "https://x.test/legal/notes.html",
    ]
    kept = filter_document_links(links, patterns=(".pdf",), required=("annual",))
    assert kept == ["https://x.test/legal/annual-report-fy2025.pdf"]


def test_resolve_listing_returns_candidates_and_errors() -> None:
    page = '<html><body><a href="/files/ar-fy2025.pdf">AR</a><a href="/files/logo.png">logo</a></body></html>'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, request=request)
        return httpx.Response(
            200, text=page, headers={"content-type": "text/html"}, request=request
        )

    p = policy()
    candidates, error = resolve_listing(
        "https://x.test/ir", policy=p, client=make_client(p, handler)
    )
    assert error is None
    assert candidates == ["https://x.test/files/ar-fy2025.pdf"]

    failing = make_client(p, lambda request: httpx.Response(500, request=request))
    candidates, error = resolve_listing("https://x.test/ir", policy=p, client=failing)
    assert candidates == []
    assert error


def test_listing_candidates_are_never_downloaded_automatically(records, tmp_path: Path) -> None:
    """A listing row must stay planned until a human names the exact file."""
    listing = make_record(
        doc_id="sbi_ar_fy2025",
        source="bank_ir",
        issuer="SBI",
        doc_type="annual_report",
        fiscal_period="FY2025",
        discovery="listing",
        url="https://sbi.co.in/web/investor-relations",
    )
    actions = plan_downloads([listing], raw_dir=tmp_path)
    # It is planned for download (the landing page), but the row keeps discovery=listing
    # so the ingest step knows the URL is not itself the document.
    assert actions[0].kind == "download"
    assert listing.discovery == "listing"
