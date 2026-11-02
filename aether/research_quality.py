"""Evidence-oriented research extensions: safe page ingestion, scoring and review records.

Scores are transparent triage heuristics. They are not truth probabilities or peer-review guarantees.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from html.parser import HTMLParser
from ipaddress import ip_address
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4
import json
import socket
import re

from aether.research import ResearchError, ResearchSource, html_to_text
from aether.storage import Store


MAX_DOCUMENT_BYTES = 2_000_000


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_public_url(url: str) -> str:
    """Reject non-HTTP URLs, credentials, localhost and non-public DNS targets."""
    parsed = urlsplit(url.strip())
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS source URLs are allowed")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Source URL must have a public hostname and no embedded credentials")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("Non-standard source URL ports are not allowed")
    host = parsed.hostname.rstrip(".").casefold()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        raise ValueError("Localhost source URLs are not allowed")
    try:
        addresses = {ip_address(host)}
    except ValueError:
        try:
            addresses = {
                ip_address(item[4][0])
                for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
            }
        except OSError as exc:
            raise ResearchError(f"Could not resolve source hostname: {exc}") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Source hostname must resolve only to public IP addresses")
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc, parsed.path or "/", parsed.query, ""))


@dataclass(frozen=True)
class QualityAssessment:
    score: int
    band: str
    factors: dict[str, int]
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchWorkflow:
    """Bounded research utilities built on AETHER's existing report store."""

    def __init__(self, store: Store, timeout_seconds: int = 15):
        self.store = store
        self.timeout_seconds = max(1, min(int(timeout_seconds), 30))
        self._opener = build_opener(_NoRedirect())

    def ingest_url(self, url: str) -> ResearchSource:
        safe_url = validate_public_url(url)
        request = Request(
            safe_url,
            headers={
                "User-Agent": "AETHER-Research/0.4 (+local evidence capture)",
                "Accept": "text/html,application/xhtml+xml,text/plain,application/pdf;q=0.5",
            },
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                content_type = response.headers.get_content_type().casefold()
                if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                    raise ResearchError(f"Unsupported source content type: {content_type}")
                payload = response.read(MAX_DOCUMENT_BYTES + 1)
                if len(payload) > MAX_DOCUMENT_BYTES:
                    raise ResearchError("Source document exceeded the 2 MB safety limit")
                charset = response.headers.get_content_charset() or "utf-8"
                body = payload.decode(charset, errors="replace")
                final_url = validate_public_url(response.geturl())
        except HTTPError as exc:
            if 300 <= exc.code < 400:
                raise ResearchError("Redirects are not followed; validate and ingest the destination URL separately") from exc
            raise ResearchError(f"Source returned HTTP {exc.code}") from exc
        except (OSError, TimeoutError, URLError, UnicodeError) as exc:
            raise ResearchError(f"Could not ingest source: {exc}") from exc

        text = html_to_text(body) if content_type != "text/plain" else " ".join(body.split())
        title_match = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
        title = html_to_text(title_match.group(1))[:500] if title_match else urlsplit(final_url).hostname or "Web document"
        if not text.strip():
            raise ResearchError("Source contained no extractable text")
        host = urlsplit(final_url).hostname or "unknown publisher"
        return ResearchSource(
            title=title or host,
            url=final_url,
            publisher=host,
            published="Date unavailable",
            evidence_type="full-text-web-document",
            summary=text[:12000],
            full_text=text,
        )

    @staticmethod
    def assess_source(source: ResearchSource) -> QualityAssessment:
        factors: dict[str, int] = {}
        caveats: list[str] = []
        evidence = source.evidence_type.casefold()
        if "peer-reviewed" in evidence or "journal" in evidence:
            factors["evidence_type"] = 25
        elif "scholarly-index" in evidence:
            factors["evidence_type"] = 16
            caveats.append("Index metadata does not prove peer review or article quality.")
        elif "preprint" in evidence or "arxiv" in evidence:
            factors["evidence_type"] = 12
            caveats.append("Preprints may not have completed peer review.")
        elif "technical-documentation" in evidence or "official" in evidence:
            factors["evidence_type"] = 18
            caveats.append("Official documentation is authoritative for the publisher's own product, not independent evaluation.")
        elif "full-text-web" in evidence:
            factors["evidence_type"] = 8
            caveats.append("General web content requires independent corroboration.")
        elif "general-web-search" in evidence:
            factors["evidence_type"] = 5
            caveats.append("Search-result metadata/snippets are leads, not inspected primary evidence.")
        elif "tertiary" in evidence or "wikipedia" in evidence:
            factors["evidence_type"] = 5
            caveats.append("Tertiary source; follow its references to primary sources.")
        else:
            factors["evidence_type"] = 6
            caveats.append("Evidence class is unspecified or not independently characterized.")

        host = (urlsplit(source.url).hostname or "").casefold()
        trusted_suffixes = (".gov", ".gov.uk", ".edu")
        if host.endswith(trusted_suffixes):
            factors["publisher_domain"] = 15
        elif host in {"doi.org", "arxiv.org", "api.crossref.org"} or host.endswith(".nih.gov"):
            factors["publisher_domain"] = 14
        else:
            factors["publisher_domain"] = 7
            caveats.append("Publisher domain alone does not establish editorial quality or accuracy.")

        factors["metadata_completeness"] = (
            10 * bool(source.title.strip())
            + 8 * bool(source.url.strip())
            + 5 * (source.published != "Date unavailable")
            + 7 * bool(source.publisher.strip())
        )
        factors["content_availability"] = 15 if len(source.summary.strip()) >= 500 else 8 if source.summary.strip() else 0
        if len(source.summary.strip()) < 500:
            caveats.append("Only a short summary/abstract is available; methods and limitations may be missing.")

        score = min(100, sum(factors.values()))
        band = "high-triage" if score >= 70 else "medium-triage" if score >= 45 else "low-triage"
        caveats.append("This score ranks review priority; it is not a probability that the source or its claims are true.")
        return QualityAssessment(score, band, factors, tuple(caveats))

    def verify_citations(self, report_id: str) -> dict[str, Any]:
        report = self.store.get_research_report(report_id)
        if report is None:
            raise ValueError(f"Unknown research report: {report_id}")
        results = []
        for source in report["sources"]:
            url = source.get("url", "")
            item = {"url": url, "listed_in_report": bool(url), "reachable": False, "status": "unverified"}
            try:
                safe_url = validate_public_url(url)
                request = Request(safe_url, headers={"User-Agent": "AETHER-Research/0.4", "Range": "bytes=0-0"})
                with self._opener.open(request, timeout=self.timeout_seconds) as response:
                    item["http_status"] = response.status
                    item["reachable"] = 200 <= response.status < 400
                    item["status"] = "reachable_only"
            except HTTPError as exc:
                item["http_status"] = exc.code
                item["status"] = "unreachable" if exc.code >= 400 else "redirect_requires_manual_validation"
            except (ValueError, ResearchError, OSError, TimeoutError, URLError) as exc:
                item["error"] = str(exc)[:300]
                item["status"] = "invalid_or_unreachable"
            item["citation_support"] = "not_checked"
            results.append(item)
        return {
            "report_id": report_id,
            "checked": len(results),
            "results": results,
            "limitation": "URL reachability and report membership do not verify that a source supports a cited claim. Claim-level support requires human/source review.",
        }

    def create_review_protocol(self, query: str, sources: list[ResearchSource]) -> dict[str, Any]:
        """Persist a reproducible evidence inventory; human screening remains explicit."""
        if len(query.strip()) < 3:
            raise ValueError("Review question must contain at least three characters")
        review_id = uuid4().hex
        assessments = []
        for source in sources:
            quality = self.assess_source(source)
            assessments.append({
                "source": source.to_dict(),
                "quality_assessment": quality.to_dict(),
                "screening_status": "awaiting_human_screen",
                "exclusion_reason": None,
            })
        content = "\n".join([
            f"# Evidence review protocol: {query}",
            "",
            f"- Review ID: {review_id}",
            f"- Created: {date.today().isoformat()}",
            "- Search sources: Crossref, arXiv, Wikipedia and manually ingested URLs as supplied",
            "- Inclusion/exclusion criteria: not yet specified; no source is silently excluded",
            "- Screening: awaiting human review",
            "- Quality scores: deterministic triage heuristics, not truth probabilities",
            "- Synthesis: not performed; evidence inventory only",
            "",
            "## Required next steps",
            "1. Define inclusion and exclusion criteria before screening.",
            "2. Screen titles/abstracts, recording a reason for every exclusion.",
            "3. Retrieve and inspect full texts where legally and technically available.",
            "4. Extract claim-level evidence with page/section locations.",
            "5. Assess risk of bias using a domain-appropriate method.",
            "6. Synthesize only after screening; report disagreements and limitations.",
        ])
        source_data = [{
            **entry["source"],
            "quality_assessment": entry["quality_assessment"],
            "screening_status": entry["screening_status"],
            "exclusion_reason": entry["exclusion_reason"],
        } for entry in assessments]
        self.store.save_research_report(review_id, f"evidence-review: {query}", content, source_data)
        return {"id": review_id, "query": query, "content": content, "sources": source_data, "screening": "pending"}
