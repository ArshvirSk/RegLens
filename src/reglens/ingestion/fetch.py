"""Manifest-driven, polite document fetcher.

Design constraints that shaped this module:

* **Respect sources.** Only public documents. ``robots.txt`` is fetched and obeyed per
  host, requests are throttled per host, and nothing is downloaded unless the operator
  explicitly confirms they have checked the site's terms of use (``--accept-terms``).
  ``--dry-run`` is the default, so nothing leaves the machine by accident.
* **Immutable raw store.** Files are stored content-addressed
  (``data/raw/<source>/<doc_id>/<doc_id>__<hash12>.pdf``) and never overwritten. If a
  re-fetch yields different bytes, both versions exist and the manifest records which
  hash is current.
* **Idempotent by hash.** A document whose recorded hash matches the file on disk is
  skipped without a network request at all, so re-running is free.
* **Listing-page discovery.** Bank investor-relations pages are listing pages, not single
  PDFs. ``resolve_listing`` extracts candidate PDF links for human review rather than
  guessing, because filenames and page structure change without warning.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from reglens.config import Settings, get_settings
from reglens.ingestion.manifest import (
    DocumentRecord,
    raw_relative_path,
    sha256_bytes,
    sha256_file,
    write_manifest,
)

ActionKind = Literal["skip_local", "download", "blocked_robots", "excluded", "error"]


@dataclass(frozen=True)
class FetchPolicy:
    """Politeness and safety limits for outbound requests."""

    user_agent: str = "RegLens/0.1 (research; contact: you@example.com)"
    delay_seconds: float = 2.0
    timeout_seconds: float = 60.0
    max_bytes: int = 50 * 1024 * 1024
    respect_robots: bool = True
    max_attempts: int = 3
    #: Some hosts only serve documents when the request looks like navigation from their
    #: own pages. rbidocs.rbi.org.in returns an HTML interstitial instead of the PDF
    #: without it (verified 2026-10-02), so the referer is derived per host.
    referer_hosts: tuple[tuple[str, str], ...] = (("rbidocs.rbi.org.in", "https://m.rbi.org.in/"),)

    @classmethod
    def from_settings(cls, settings: Settings) -> FetchPolicy:
        return cls(
            user_agent=settings.user_agent,
            delay_seconds=settings.fetch_delay_seconds,
            timeout_seconds=settings.fetch_timeout_seconds,
            max_bytes=settings.fetch_max_bytes,
            respect_robots=settings.fetch_respect_robots,
        )


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content: bytes
    content_type: str
    elapsed_ms: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status_code < 300

    @property
    def suffix(self) -> str:
        path_suffix = Path(urlparse(self.final_url).path).suffix.lower()
        if path_suffix in {".pdf", ".html", ".htm", ".txt", ".csv", ".xlsx", ".json"}:
            return path_suffix
        return ".html" if "html" in self.content_type else ".bin"


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"a", "link"}:
            return
        for name, value in attrs:
            if name == "href" and value:
                self.hrefs.append(value)


class RobotsCache:
    """Per-host ``robots.txt`` decisions, cached for the lifetime of a run."""

    def __init__(self, policy: FetchPolicy) -> None:
        self._policy = policy
        self._cache: dict[str, RobotFileParser | None] = {}

    def _parser(self, client: httpx.Client, url: str) -> RobotFileParser | None:
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin in self._cache:
            return self._cache[origin]
        parser: RobotFileParser | None = RobotFileParser()
        try:
            response = client.get(f"{origin}/robots.txt", follow_redirects=True)
        except httpx.HTTPError:
            # Unreachable robots.txt is treated as "no rules published"; the per-host
            # delay still applies. Documented rather than silent.
            parser = None
        else:
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                parser = None
        self._cache[origin] = parser
        return parser

    def allowed(self, client: httpx.Client, url: str) -> tuple[bool, str]:
        """Return ``(allowed, reason)``. Unknown/unreachable rules allow, with a reason."""
        if not self._policy.respect_robots:
            return True, "robots checking disabled by policy"
        parser = self._parser(client, url)
        if parser is None:
            return True, "robots.txt unavailable or unpublished"
        if parser.can_fetch(self._policy.user_agent, url):
            return True, "allowed by robots.txt"
        return False, "disallowed by robots.txt"


@dataclass
class _Throttle:
    delay_seconds: float
    _last: dict[str, float] = field(default_factory=dict)

    def wait(self, url: str) -> None:
        host = urlparse(url).netloc
        last = self._last.get(host)
        now = time.monotonic()
        if last is not None:
            remaining = self.delay_seconds - (now - last)
            if remaining > 0:
                time.sleep(remaining)
        self._last[host] = time.monotonic()


class FetchError(RuntimeError):
    """Raised for transport-level failures that retrying could not fix."""


def referer_for(url: str, policy: FetchPolicy) -> str | None:
    """The referer a host expects, if it is known to require one."""
    host = urlparse(url).netloc.lower()
    for pattern, referer in policy.referer_hosts:
        if host == pattern or host.endswith("." + pattern):
            return referer
    return None


def build_client(policy: FetchPolicy) -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": policy.user_agent,
            "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.5",
        },
        timeout=policy.timeout_seconds,
        follow_redirects=True,
    )


def _get_with_retry(
    client: httpx.Client,
    url: str,
    policy: FetchPolicy,
    *,
    headers: dict[str, str] | None = None,
) -> FetchResult:
    """Retry transient failures (timeouts, 429s, 5xx) with exponential backoff."""
    retryer = Retrying(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(policy.max_attempts),
        wait=wait_exponential(multiplier=1.5, min=1, max=20),
        reraise=True,
    )
    return retryer(_get, client, url, policy, headers)


def _get(
    client: httpx.Client,
    url: str,
    policy: FetchPolicy,
    headers: dict[str, str] | None = None,
) -> FetchResult:
    started = time.monotonic()
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code in {408, 425, 429, 500, 502, 503, 504}:
            raise httpx.HTTPStatusError(
                f"retryable status {response.status_code}",
                request=response.request,
                response=response,
            )
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > policy.max_bytes:
            raise FetchError(f"content-length {declared} exceeds max_bytes {policy.max_bytes}")
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > policy.max_bytes:
                raise FetchError(f"response exceeded max_bytes {policy.max_bytes}")
            chunks.append(chunk)
        content = b"".join(chunks)
        return FetchResult(
            url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            content=content,
            content_type=response.headers.get("content-type", "").split(";")[0].strip(),
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )


def fetch_bytes(
    url: str,
    *,
    client: httpx.Client,
    policy: FetchPolicy,
    robots: RobotsCache,
    throttle: _Throttle,
) -> FetchResult:
    """One throttled, robots-checked, size-capped, retrying GET."""
    allowed, _reason = robots.allowed(client, url)
    if not allowed:
        return FetchResult(url, url, 0, b"", "", 0, error="blocked by robots.txt")
    throttle.wait(url)
    referer = referer_for(url, policy)
    headers = {"Referer": referer} if referer else None
    try:
        return _get_with_retry(client, url, policy, headers=headers)
    except httpx.HTTPStatusError as exc:
        return FetchResult(url, url, exc.response.status_code, b"", "", 0, error=str(exc))
    except (httpx.HTTPError, FetchError) as exc:
        return FetchResult(url, url, 0, b"", "", 0, error=str(exc))


# ------------------------------------------------------------------ planning
@dataclass(frozen=True)
class DownloadAction:
    doc_id: str
    url: str
    kind: ActionKind
    reason: str


@dataclass
class DownloadReport:
    actions: list[DownloadAction] = field(default_factory=list)
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    new_rows: list[DocumentRecord] = field(default_factory=list)
    dry_run: bool = True

    def counts(self) -> dict[str, int]:
        """Plan counts by action kind, plus what this run actually did.

        An ``excluded`` row is not a failure, and a ``blocked_robots`` row is not a bug:
        keeping them as separate buckets stops a polite skip from looking like an error.
        """
        summary: dict[str, int] = {"total": len(self.actions)}
        for action in self.actions:
            summary[action.kind] = summary.get(action.kind, 0) + 1
        summary["downloaded_now"] = len(self.downloaded)
        summary["failed_now"] = len(self.failures)
        return summary


def plan_downloads(
    records: Iterable[DocumentRecord],
    *,
    raw_dir: Path,
    only: set[str] | None = None,
    limit: int | None = None,
) -> list[DownloadAction]:
    """Decide what each row needs, without any network access."""
    actions: list[DownloadAction] = []
    for record in records:
        if only and record.doc_id not in only:
            continue
        if record.status == "excluded":
            actions.append(
                DownloadAction(record.doc_id, record.url, "excluded", "excluded in manifest")
            )
            continue
        if record.file_hash and record.local_path:
            path = raw_dir / record.local_path
            if path.is_file() and sha256_file(path) == record.file_hash:
                actions.append(
                    DownloadAction(
                        record.doc_id, record.url, "skip_local", "already fetched and hash matches"
                    )
                )
                continue
            actions.append(
                DownloadAction(
                    record.doc_id, record.url, "download", "local file missing or hash mismatch"
                )
            )
            continue
        if not record.url:
            actions.append(DownloadAction(record.doc_id, record.url, "error", "row has no url"))
            continue
        actions.append(DownloadAction(record.doc_id, record.url, "download", "not fetched yet"))
    if limit is not None:
        kept: list[DownloadAction] = []
        to_download = 0
        for action in actions:
            if action.kind == "download":
                to_download += 1
                if to_download > limit:
                    continue
            kept.append(action)
        actions = kept
    return actions


def download_manifest(
    records: list[DocumentRecord],
    *,
    policy: FetchPolicy,
    settings: Settings | None = None,
    dry_run: bool = True,
    accept_terms: bool = False,
    only: set[str] | None = None,
    limit: int | None = None,
    client: httpx.Client | None = None,
) -> DownloadReport:
    """Download every planned document, then rewrite the manifest with the results.

    ``client`` is injectable so the whole path can be tested against a mock transport.
    """
    active = settings or get_settings()
    if not dry_run and not accept_terms:
        raise PermissionError(
            "refusing to download: pass accept_terms=True only after checking the source "
            "site's terms of use and robots.txt (CLI: --accept-terms)"
        )

    actions = plan_downloads(records, raw_dir=active.raw_dir, only=only, limit=limit)
    report = DownloadReport(actions=actions, dry_run=dry_run)
    by_id = {record.doc_id: index for index, record in enumerate(records)}
    robots = RobotsCache(policy)
    throttle = _Throttle(policy.delay_seconds)
    owns_client = client is None
    active_client = client or build_client(policy)

    try:
        for action in actions:
            if action.kind != "download" or dry_run:
                continue
            record = records[by_id[action.doc_id]]
            result = fetch_bytes(
                record.url, client=active_client, policy=policy, robots=robots, throttle=throttle
            )
            if not result.ok:
                report.failures.append(
                    (record.doc_id, result.error or f"HTTP {result.status_code}")
                )
                record.status = "failed"
                record.http_status = str(result.status_code)
                record.notes = _append_note(record.notes, f"fetch failed: {result.error}")
                continue
            digest = sha256_bytes(result.content)
            suffix = result.suffix
            relative = raw_relative_path(record, digest, suffix)
            target = active.raw_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(result.content)
            updated = record.model_copy(
                update={
                    "status": "fetched",
                    "file_hash": digest,
                    "local_path": relative,
                    "http_status": str(result.status_code),
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
            records[by_id[record.doc_id]] = updated
            report.downloaded.append(record.doc_id)
    finally:
        if owns_client:
            active_client.close()

    if not dry_run:
        write_manifest(records, active.manifest_path)
    return report


def _append_note(existing: str, note: str) -> str:
    return f"{existing} | {note}" if existing else note


# ------------------------------------------------------------------ listing pages
def extract_links(html: str, base_url: str) -> list[str]:
    """All absolute links on a page, in document order, de-duplicated."""
    parser = _LinkExtractor()
    parser.feed(html)
    seen: dict[str, None] = {}
    for href in parser.hrefs:
        absolute = urljoin(base_url, href)
        if absolute.startswith(("http://", "https://")):
            seen.setdefault(absolute, None)
    return list(seen)


def filter_document_links(
    links: Iterable[str],
    *,
    patterns: Iterable[str] = (".pdf",),
    required: Iterable[str] = (),
) -> list[str]:
    """Keep links whose path matches a pattern and contains every required substring."""
    wanted = tuple(item.lower() for item in patterns)
    musts = tuple(item.lower() for item in required)
    kept: list[str] = []
    for link in links:
        path = urlparse(link).path.lower()
        if wanted and not any(pattern in path for pattern in wanted):
            continue
        if musts and not all(must in path for must in musts):
            continue
        kept.append(link)
    return kept


def resolve_listing(
    url: str,
    *,
    policy: FetchPolicy,
    patterns: Iterable[str] = (".pdf",),
    required: Iterable[str] = (),
    client: httpx.Client | None = None,
) -> tuple[list[str], str | None]:
    """Fetch a listing page and return candidate document URLs for human review.

    Returns ``(candidates, error)``. Candidates are never downloaded automatically: the
    operator adds the ones they want to ``data/corpus_plan.yaml`` first, which keeps the
    corpus an explicit, reviewable decision instead of whatever a page happened to link.
    """
    robots = RobotsCache(policy)
    throttle = _Throttle(policy.delay_seconds)
    owns_client = client is None
    active_client = client or build_client(policy)
    try:
        result = fetch_bytes(
            url, client=active_client, policy=policy, robots=robots, throttle=throttle
        )
    finally:
        if owns_client:
            active_client.close()
    if not result.ok:
        return [], result.error or f"HTTP {result.status_code}"
    html = result.content.decode("utf-8", errors="replace")
    candidates = filter_document_links(
        extract_links(html, result.final_url), patterns=patterns, required=required
    )
    return candidates, None
