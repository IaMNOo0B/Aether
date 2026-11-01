"""Bounded public-source research using Crossref, arXiv, and Wikipedia APIs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4
import json
import logging
import re
import xml.etree.ElementTree as ET

from aether.storage import Store

logger = logging.getLogger(__name__)


class ResearchError(RuntimeError):
    pass


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


def html_to_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    return unescape(" ".join(parser.parts))


@dataclass(frozen=True)
class ResearchSource:
    title: str
    url: str
    publisher: str
    published: str
    evidence_type: str
    summary: str
    full_text: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)



class _SearchResultParser(HTMLParser):
    """Extract result links/snippets from DuckDuckGo HTML results."""
    def __init__(self) -> None:
        super().__init__()
        self.results = []
        self.current = None
        self.capture = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self.current = {"title": "", "url": values.get("href") or "", "snippet": ""}
            self.capture = "title"
        elif self.current is not None and tag in {"a", "div", "td"} and "result__snippet" in classes:
            self.capture = "snippet"

    def handle_data(self, data):
        if self.current is not None and self.capture:
            self.current[self.capture] += data

    def handle_endtag(self, tag):
        if self.current is not None and tag == "a" and self.capture == "title":
            self.capture = None
        if self.current is not None and tag in {"div", "td"} and self.capture == "snippet":
            self.capture = None
        if self.current is not None and tag == "div" and self.current["title"].strip():
            if self.current["url"].strip():
                self.results.append({key: " ".join(value.split()) for key, value in self.current.items()})
                self.current = None
                self.capture = None


class Researcher:
    """Research adapter; fetched pages are data, never executable instructions."""

    def __init__(self, store: Store, timeout_seconds: int = 15, max_results: int = 8):
        self.store = store
        self.timeout_seconds = timeout_seconds
        self.max_results = max(1, min(max_results, 20))

    def _get(self, url: str) -> bytes:
        request = Request(
            url,
            headers={
                "User-Agent": "AETHER-Research/0.3 (local research assistant)",
                "Accept": "application/json, application/atom+xml, application/xml, text/html",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read(2_000_001)
                if len(payload) > 2_000_000:
                    raise ResearchError("Research response exceeded the 2 MB safety limit")
                return payload
        except (OSError, TimeoutError, HTTPError, URLError) as exc:
            raise ResearchError(f"Could not retrieve research source: {exc}") from exc

    def search_crossref(self, query: str, limit: int = 5) -> list[ResearchSource]:
        params = urlencode({"query.bibliographic": query, "rows": min(limit, 10), "select": "title,author,published,created,container-title,URL,DOI,abstract"})
        payload = json.loads(self._get("https://api.crossref.org/works?" + params))
        items = payload.get("message", {}).get("items", [])
        sources = []
        for item in items:
            title = (item.get("title") or ["Untitled scholarly work"])[0]
            authors = item.get("author") or []
            author_text = ", ".join(
                " ".join(part for part in (author.get("given"), author.get("family")) if part)
                for author in authors[:3]
            )
            container = (item.get("container-title") or ["Crossref"])[0]
            date_parts = (
                item.get("published", {}).get("date-parts")
                or item.get("created", {}).get("date-parts")
                or [[]]
            )[0]
            published = "-".join(str(part) for part in date_parts) if date_parts else "Date unavailable"
            abstract = html_to_text(item.get("abstract", ""))
            summary = abstract[:1400] or f"{author_text}. Scholarly record indexed by Crossref."
            sources.append(ResearchSource(
                title=str(title), url=str(item.get("URL") or ""),
                publisher=str(container), published=published,
                evidence_type="scholarly-index-record", summary=summary,
            ))
        return sources

    def search_arxiv(self, query: str, limit: int = 5) -> list[ResearchSource]:
        params = urlencode({"search_query": f"all:{query}", "start": 0, "max_results": min(limit, 10)})
        root = ET.fromstring(self._get("https://export.arxiv.org/api/query?" + params))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        sources = []
        for entry in root.findall("a:entry", ns):
            title = " ".join((entry.findtext("a:title", default="", namespaces=ns)).split())
            summary = " ".join((entry.findtext("a:summary", default="", namespaces=ns)).split())
            link = next(
                (item.attrib.get("href", "") for item in entry.findall("a:link", ns)
                 if item.attrib.get("rel") == "alternate"),
                "",
            )
            published = entry.findtext("a:published", default="Date unavailable", namespaces=ns)[:10]
            sources.append(ResearchSource(
                title=title or "Untitled arXiv preprint", url=link,
                publisher="arXiv", published=published,
                evidence_type="preprint", summary=summary[:1400],
            ))
        return sources

    def search_wikipedia(self, query: str, limit: int = 3) -> list[ResearchSource]:
        params = urlencode({
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": min(limit, 5), "format": "json",
        })
        data = json.loads(self._get("https://en.wikipedia.org/w/api.php?" + params))
        results = []
        for item in data.get("query", {}).get("search", []):
            title = str(item.get("title", ""))
            page = json.loads(self._get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title.replace(" ", "_"))
            ))
            results.append(ResearchSource(
                title=title,
                url=str(page.get("content_urls", {}).get("desktop", {}).get("page")
                        or "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))),
                publisher="Wikipedia",
                published=str(page.get("timestamp", "Date unavailable"))[:10],
                evidence_type="tertiary-reference",
                summary=str(page.get("extract", item.get("snippet", "")))[:1400],
            ))
        return results


    def search_web(self, query: str, limit: int = 5) -> list[ResearchSource]:
        """Search public web results; result pages are not fetched or executed."""
        text = query.strip()
        if len(text) < 3 or len(text) > 500:
            raise ValueError("Web search query must contain 3..500 characters")
        params = urlencode({"q": text, "kl": "wt-wt"})
        payload = self._get("https://html.duckduckgo.com/html/?" + params)
        parser = _SearchResultParser()
        parser.feed(payload.decode("utf-8", errors="replace"))
        results = []
        for item in parser.results:
            raw_url = item["url"]
            parsed = urlsplit(raw_url)
            if parsed.hostname and parsed.hostname.casefold().endswith("duckduckgo.com"):
                destination = parse_qs(parsed.query).get("uddg", [""])[0]
                if destination:
                    raw_url = unquote(destination)
            target = urlsplit(raw_url)
            if target.scheme not in {"http", "https"} or not target.hostname:
                continue
            host = target.hostname.casefold()
            if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".localhost"):
                continue
            title = item["title"][:500]
            if not title:
                continue
            results.append(ResearchSource(
                title=title, url=raw_url, publisher=host,
                published="Date unavailable",
                evidence_type="general-web-search-result",
                summary=item["snippet"][:1400] or "Search result snippet unavailable; inspect the source before use.",
            ))
            if len(results) >= max(1, min(limit, 10)):
                break
        return results

    def search(self, query: str, limit: int | None = None) -> list[ResearchSource]:
        text = query.strip()
        if len(text) < 3:
            raise ValueError("Research query must contain at least three characters")
        if len(text) > 500:
            raise ValueError("Research query is limited to 500 characters")
        count = max(1, min(limit or self.max_results, self.max_results))
        provider_results: list[list[ResearchSource]] = []
        errors: list[str] = []
        successful_providers = 0
        for label, operation, amount in (
            ("Web", self.search_web, min(4, count)),
            ("Crossref", self.search_crossref, max(2, count // 2)),
            ("arXiv", self.search_arxiv, max(2, count // 2)),
            ("Wikipedia", self.search_wikipedia, min(2, count)),
        ):
            try:
                provider_results.append(operation(text, amount))
                successful_providers += 1
            except ResearchError as exc:
                errors.append(f"{label}: {exc}")
                logger.warning("%s search failed: %s", label, exc)
                provider_results.append([])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError, ET.ParseError) as exc:
                errors.append(f"{label}: invalid response ({exc})")
                logger.warning("%s returned an invalid response: %s", label, exc)
                provider_results.append([])

        # Interleave providers so the general web adapter cannot crowd out scholarly sources.
        unique: dict[str, ResearchSource] = {}
        max_provider_results = max((len(items) for items in provider_results), default=0)
        for index in range(max_provider_results):
            for items in provider_results:
                if index >= len(items):
                    continue
                source = items[index]
                key = source.url or source.title.casefold()
                unique.setdefault(key, source)
        results = list(unique.values())[:count]
        if not results and errors and successful_providers == 0:
            raise ResearchError("All research providers failed: " + "; ".join(errors))
        return results

    def create_report(self, query: str, sources: list[ResearchSource]) -> dict[str, Any]:
        report_id = uuid4().hex
        sections = [
            f"# Research report: {query}",
            "",
            "## Findings",
            "",
            "This report summarizes indexed source records. It is not a systematic review.",
            "Preprints are not necessarily peer reviewed. Wikipedia is a tertiary starting point.",
            "",
        ]
        for index, source in enumerate(sources, start=1):
            sections.extend([
                f"### [{index}] {source.title}",
                f"- Publisher: {source.publisher}",
                f"- Date: {source.published}",
                f"- Evidence class: {source.evidence_type}",
                f"- Source: {source.url}",
                "",
                source.summary or "No abstract or summary was supplied by the index.",
                "",
            ])
        if not sources:
            sections.extend(["No sources were returned. Do not infer that no research exists.", ""])
        sections.extend([
            "## Limitations",
            "",
            "- Search ranking is not a quality assessment.",
            "- Abstracts and summaries may omit important methods or limitations.",
            "- Verify primary papers and publication status before relying on a claim.",
            "- This report does not establish causation or consensus.",
        ])
        report = "\n".join(sections)
        source_data = [source.to_dict() for source in sources]
        self.store.save_research_report(report_id, query, report, source_data)
        return {"id": report_id, "query": query, "content": report, "sources": source_data}
