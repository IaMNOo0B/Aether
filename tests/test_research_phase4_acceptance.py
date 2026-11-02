"""Deterministic synthetic end-to-end acceptance fixture for Phase 4."""
from aether.research import ResearchSource
from aether.research_advanced import AdvancedResearch
from aether.research_completion import GRADE_DOMAINS, RO_B_DOMAINS, ResearchCompletion
from aether.storage import Store


def test_phase4_end_to_end_synthetic_acceptance(tmp_path):
    store = Store(tmp_path / "phase4.sqlite3")
    completion = ResearchCompletion(store)
    credentials = {
        "reviewer-a": "correct horse battery A",
        "reviewer-b": "correct horse battery B",
        "adjudicator": "correct horse battery C",
    }
    for reviewer_id, password in credentials.items():
        completion.register_reviewer(reviewer_id, password)

    research = AdvancedResearch(store)
    review = research.create_systematic_review(
        "Does intervention X improve response time in adults?",
        ["Adult human participants", "Reports response-time outcome"],
        ["Animal studies", "No response-time outcome"],
        {"databases": ["Synthetic fixture"], "queries": ["intervention X response time"],
         "searched_on": "2026-11-02"},
    )
    sources = [
        ResearchSource(
            title=f"Synthetic study {index}", url=f"https://fixture.example/study-{index}",
            publisher="Synthetic Evidence Lab", published="2026",
            evidence_type="synthetic-rct", summary="Adult participants; response-time outcome.",
            full_text=f"The intervention reduced mean response time in the adult study group. Mean difference was {-3.0 * index} ms with standard error 1.5.",
        )
        for index in (1, 2)
    ]
    research.add_review_sources(review["id"], sources)

    for source in sources:
        for reviewer_id, password in credentials.items():
            if reviewer_id == "adjudicator":
                continue
            research.screen_source(
                review["id"], source.url, reviewer_id, "include",
                "The synthetic abstract meets both prespecified eligibility criteria.",
                password=password,
            )
    research.finalize_review(review["id"])

    randomized_domains = RO_B_DOMAINS["randomized"]
    for index, source in enumerate(sources, 1):
        full_text = {
            "decision": "include",
            "rationale": "Full text confirms the adult population and response-time outcome.",
            "text_locator": "Results, paragraph 2",
        }
        ratings = {
            domain: {"rating": "low", "rationale": "Synthetic report gives sufficient detail for assessment."}
            for domain in randomized_domains
        }
        extraction = {
            "study_design": "randomized controlled trial",
            "population": "Adults with condition X",
            "sample_size": 100 + index,
            "outcome": "response time",
            "findings": "Intervention reduced response time.",
            "effect_measure": "mean_difference_ms",
            "effect_estimate": -3.0 * index,
            "standard_error": 1.5,
            "effect_evidence_quote": f"Mean difference was {-3.0 * index} ms with standard error 1.5.",
            "effect_text_locator": "Results, Table 2",
        }
        for reviewer_id, password in credentials.items():
            if reviewer_id == "adjudicator":
                continue
            research.record_review_stage(
                review["id"], source.url, "full_text", reviewer_id, full_text, password=password,
            )
            research.record_review_stage(
                review["id"], source.url, "risk_of_bias", reviewer_id,
                {"design": "randomized", "instrument": "AETHER-structured-randomized-v1",
                 "domains": ratings, "limitation": "Synthetic fixture; not a validated instrument."},
                password=password,
            )
            research.record_review_stage(
                review["id"], source.url, "extraction", reviewer_id, extraction, password=password,
            )

    synthesized = research.synthesize_review(review["id"])
    assert synthesized["status"] == "synthesized"
    assert synthesized["final_summary"]["meta_analysis"]["studies"] == 2
    assert synthesized["final_summary"]["effect_evidence_validation"]["exact_quote_and_numeric_values_found"] == 2

    flow = completion.prisma_flow(review["id"])
    assert flow["included"]["studies_in_review"] == 2
    assert flow["included"]["studies_in_quantitative_synthesis"] == 2
    assert completion.publication_bias_for_review(review["id"])["status"] == "not_tested_fewer_than_10_studies"

    judgments = {
        domain: {"judgment": "none", "rationale": "The synthetic evidence does not justify a downgrade."}
        for domain in GRADE_DOMAINS
    }
    judgments["imprecision"] = {
        "judgment": "serious",
        "rationale": "The small synthetic sample leaves meaningful uncertainty around the effect.",
    }
    grade = completion.grade_certainty(
        review["id"], "response time", "high", judgments,
        "The synthetic evidence is consistent but remains imprecise due to small samples.",
    )
    assert grade["certainty"] == "moderate"

    store.save_research_report("citation-report", "synthetic query", "synthetic report", [{
        "title": "Synthetic citation source", "url": "https://fixture.example/citation",
        "full_text": "The intervention reduced mean response time in the adult study group.",
        "summary": "Synthetic full text for citation test.",
    }])
    evidence = research.check_citation_support(
        "The intervention reduced mean response time in adults.",
        "citation-report", "https://fixture.example/citation",
        "The intervention reduced mean response time in the adult study group.",
    )
    for reviewer_id, password in credentials.items():
        if reviewer_id == "adjudicator":
            continue
        decision = research.adjudicate_citation(
            evidence["id"], reviewer_id, "supports",
            "The exact passage states the outcome described by the claim.",
            password=password,
        )
    assert decision["human_decision"] == "supports"

    store.create_research_experiment(
        "phase4-exp", "Synthetic treatment comparison", None,
        {"objective": "Compare response time", "method": "Independent groups",
         "success_criteria": "Calculate group difference"},
    )
    store.transition_research_experiment("phase4-exp", "running")
    raw = b"group,value\ncontrol,10\ncontrol,11\ncontrol,12\ntreatment,7\ntreatment,8\ntreatment,9\n"
    observed = completion.ingest_raw_observations(
        "phase4-exp", raw, "group", "value", "treatment", "control",
    )
    assert observed["mean_difference_treatment_minus_control"] == -3
    assert store.list_research_experiment_events("phase4-exp")[-1]["event_type"] == "raw_observations_ingested"
