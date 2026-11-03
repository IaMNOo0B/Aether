"""Deterministic cognitive strategy selection with explicit prerequisites and fallbacks."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import re

from aether.model_registry import ModelCatalog


@dataclass(frozen=True)
class CognitivePlan:
    intent: str
    strategy: str
    required_capabilities: tuple[str, ...]
    needs_external_evidence: bool
    needs_human_approval: bool
    steps: tuple[str, ...]
    limitations: tuple[str, ...]


class CognitiveCoordinator:
    """Route tasks to explicit AETHER subsystems; this is a router, not a reasoning model."""

    _RULES = (
        ("high_impact", re.compile(r"\b(delete|wipe|format|transfer money|purchase|deploy to production|medical diagnosis|legal advice|security incident)\b", re.I)),
        ("multimodal_video", re.compile(r"\b(video|clip|recording|frames|scene)\b", re.I)),
        ("multimodal_audio", re.compile(r"\b(audio|speech|transcribe|voice|sound)\b", re.I)),
        ("multimodal_vision", re.compile(r"\b(image|photo|picture|screenshot|visual)\b", re.I)),
        ("symbolic_reasoning", re.compile(r"\b(logic|truth table|prove|proposition|syllogism)\b", re.I)),
        ("mathematical_reasoning", re.compile(r"\b(solve|equation|matrix|linear system|calculate|integral|derivative)\b", re.I)),
        ("research", re.compile(r"\b(research|latest|current|recent|sources|citations|systematic review|paper)\b", re.I)),
        ("planning", re.compile(r"\b(plan|implement|build|execute|roadmap|multi.step|workflow|project)\b", re.I)),
        ("code_engineering", re.compile(r"\b(code|bug|error|test|debug|refactor|repository|compile)\b", re.I)),
        ("comparison", re.compile(r"\b(compare|versus|vs\.?|trade.off|alternatives|hypotheses)\b", re.I)),
    )

    def __init__(self, model_catalog: ModelCatalog | None = None):
        self.model_catalog = model_catalog

    def route(self, task: str, *, privacy: str = "confidential",
              context_window: int = 1, max_vram_mb: int | None = None) -> dict[str, Any]:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task text is required")
        intent = "direct"
        for candidate, pattern in self._RULES:
            if pattern.search(task):
                intent = candidate
                break
        plans = {
            "high_impact": CognitivePlan(
                intent, "human_review_and_policy_gate", ("text",), False, True,
                ("Clarify the requested outcome", "Check permission and safety policy",
                 "Require explicit approval before any consequential action"),
                ("The router cannot replace domain-specific professional review.",)),
            "multimodal_video": CognitivePlan(
                intent, "video_frame_sampling_then_vision_review", ("video_frames", "vision"), False, False,
                ("Check media limits", "Extract bounded frames", "Interpret frames with a configured vision model",
                 "Keep temporal/audio limitations explicit"),
                ("Frame extraction alone is not video understanding.",)),
            "multimodal_audio": CognitivePlan(
                intent, "audio_metadata_or_local_transcription", ("audio",), False, False,
                ("Validate audio format and size", "Inspect metadata", "Use a configured local speech model if requested"),
                ("Transcription is not non-speech event classification.",)),
            "multimodal_vision": CognitivePlan(
                intent, "vision_provider_or_ocr", ("vision",), False, False,
                ("Validate image type and size", "Use a configured vision provider or OCR", "Report uncertainty"),
                ("OCR alone is not general visual understanding.",)),
            "symbolic_reasoning": CognitivePlan(
                intent, "bounded_symbolic_logic", ("symbolic_logic",), False, False,
                ("Parse only the supported logical grammar", "Evaluate the expression or truth table",
                 "Reject unsupported syntax rather than executing arbitrary code"),
                ("This route does not provide general theorem proving.",)),
            "mathematical_reasoning": CognitivePlan(
                intent, "deterministic_numerical_solver", ("numerical_reasoning",), False, False,
                ("Validate dimensions and finite inputs", "Use a bounded numerical method",
                 "Report residuals and singularity warnings"),
                ("Numerical output depends on the supplied model and assumptions.",)),
            "research": CognitivePlan(
                intent, "evidence_first_research", ("web_search", "source_provenance"), True, False,
                ("Define the research question", "Search primary and independent sources",
                 "Capture provenance", "Separate evidence from interpretation"),
                ("AETHER must run its research workflow; this route does not itself access the internet.",)),
            "planning": CognitivePlan(
                intent, "dependency_aware_plan_and_verify", ("planning", "task_receipts"), False, True,
                ("Decompose into a dependency graph", "Check resource and permission constraints",
                 "Execute only approved steps", "Verify postconditions"),
                ("The plan is not evidence that actions were executed.",)),
            "code_engineering": CognitivePlan(
                intent, "reproduce_root_cause_and_test", ("code_inspection", "sandboxed_tests"), False, True,
                ("Reproduce the failure", "Inspect logs and call paths", "Patch a candidate branch",
                 "Run baseline and candidate tests in isolation", "Require review before integration"),
                ("A passing test suite does not prove all specifications are satisfied.",)),
            "comparison": CognitivePlan(
                intent, "multi_hypothesis_evidence_comparison", ("evidence_ranking",), True, False,
                ("List competing hypotheses", "Expose assumptions", "Attach evidence and likelihood ratios",
                 "Preserve unresolved disagreement"),
                ("Conditional hypothesis scores are not calibrated probabilities.",)),
            "direct": CognitivePlan(
                intent, "direct_response", ("text",), False, False,
                ("Answer the request directly", "State uncertainty and limitations"),
                ("Complex tasks should be routed to a specialized workflow.",)),
        }
        plan = plans[intent]
        model = None
        if self.model_catalog is not None:
            model = self.model_catalog.select(
                plan.required_capabilities, privacy=privacy,
                min_context_window=context_window, max_vram_mb=max_vram_mb,
            )
        return {"plan": asdict(plan), "model_selection": model,
                "router": "deterministic-rule-based", "not_claimed": "This router does not itself execute the selected workflow."}
