import pytest

from aether.rlcd import (
    DecisionCase, DecisionPrediction, evaluate_held_out_decision_producer,
)


class TableProducer:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict(self, question, domain):
        return DecisionPrediction(self.probabilities[question], "held-out-fixture")


def test_typed_decision_producer_is_evaluated_against_held_out_outcomes():
    cases = [
        DecisionCase("1", "event one", "network", True),
        DecisionCase("2", "event two", "network", False),
        DecisionCase("3", "event three", "network", True),
        DecisionCase("4", "event four", "network", False),
    ]
    report = evaluate_held_out_decision_producer(
        cases,
        TableProducer({"event one": .9, "event two": .1, "event three": .8, "event four": .2}),
        TableProducer({"event one": .6, "event two": .4, "event three": .6, "event four": .4}),
    )
    assert report["split"] == "held_out"
    assert report["producer"]["brier_score"] < report["baseline"]["brier_score"]
    assert report["delta_brier_score"] < 0
    assert len(report["dataset_sha256"]) == 64


def test_decision_benchmark_rejects_training_split_and_invalid_predictions():
    with pytest.raises(ValueError, match="held_out"):
        evaluate_held_out_decision_producer(
            [DecisionCase("1", "question", "domain", True, split="train")],
            TableProducer({"question": .8}),
        )
    with pytest.raises(ValueError, match="finite probability"):
        evaluate_held_out_decision_producer(
            [DecisionCase("1", "question", "domain", True)],
            TableProducer({"question": float("nan")}),
        )
