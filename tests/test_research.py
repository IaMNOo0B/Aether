import json

from aether.research import Researcher
from aether.storage import Store


def test_crossref_results_are_structured_with_provenance(tmp_path):
    researcher = Researcher(Store(tmp_path / "research.sqlite3"))
    researcher._get = lambda url: json.dumps({
        "message": {"items": [{
            "title": ["A Study of Agents"],
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "container-title": ["Journal of Systems"],
            "published": {"date-parts": [[2025, 4, 2]]},
            "URL": "https://doi.org/10.1000/example",
            "abstract": "<p>We evaluate agent systems.</p>",
        }]}
    }).encode()
    results = researcher.search_crossref("agent systems", 1)
    assert results[0].title == "A Study of Agents"
    assert results[0].published == "2025-4-2"
    assert "evaluate agent systems" in results[0].summary
    assert results[0].evidence_type == "scholarly-index-record"


def test_research_report_is_persisted_with_source_records(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    researcher = Researcher(store)
    researcher.search = lambda query, limit=None: []
    report = researcher.create_report("test topic", [])
    saved = store.get_research_report(report["id"])
    assert saved["query"] == "test topic"
    assert "Limitations" in saved["content"]
    assert saved["sources"] == []
