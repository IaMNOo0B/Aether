"""Deterministic symbolic reasoning primitives and evidence-conditioned hypothesis ranking."""
from __future__ import annotations
import ast
from itertools import product
from math import isfinite, log, exp
from typing import Any, Mapping


class ReasoningInputError(ValueError):
    pass


def _evaluate_node(node: ast.AST, values: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, values)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ReasoningInputError(f"Missing proposition variable: {node.id}")
        return values[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float, str)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(_evaluate_node(node.operand, values))
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        parts = [_evaluate_node(item, values) for item in node.values]
        return all(bool(item) for item in parts) if isinstance(node.op, ast.And) else any(bool(item) for item in parts)
    if isinstance(node, ast.Compare):
        left = _evaluate_node(node.left, values)
        for operation, comparator in zip(node.ops, node.comparators):
            right = _evaluate_node(comparator, values)
            if isinstance(operation, ast.Eq): valid = left == right
            elif isinstance(operation, ast.NotEq): valid = left != right
            elif isinstance(operation, ast.Lt): valid = left < right
            elif isinstance(operation, ast.LtE): valid = left <= right
            elif isinstance(operation, ast.Gt): valid = left > right
            elif isinstance(operation, ast.GtE): valid = left >= right
            else: raise ReasoningInputError("Unsupported comparison operator")
            if not valid: return False
            left = right
        return True
    raise ReasoningInputError(f"Unsupported expression syntax: {type(node).__name__}")


def evaluate_proposition(expression: str, values: Mapping[str, Any]) -> bool:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ReasoningInputError("Invalid proposition expression") from exc
    result = _evaluate_node(tree, values)
    if not isinstance(result, bool):
        raise ReasoningInputError("Proposition expression must evaluate to a boolean")
    return result


def truth_table(expression: str, *, max_variables: int = 8) -> dict[str, Any]:
    tree = ast.parse(expression, mode="eval")
    variables = sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})
    if len(variables) > max_variables:
        raise ReasoningInputError(f"Truth table is limited to {max_variables} variables")
    rows = [
        {"values": dict(zip(variables, bits)),
         "result": evaluate_proposition(expression, dict(zip(variables, bits)))}
        for bits in product((False, True), repeat=len(variables))
    ]
    return {"expression": expression, "variables": variables, "rows": rows,
            "tautology": all(row["result"] for row in rows),
            "contradiction": not any(row["result"] for row in rows)}


def solve_linear_system(matrix: list[list[float]], vector: list[float],
                        *, epsilon: float = 1e-12) -> dict[str, Any]:
    """Solve Ax=b with partial pivoting; report residual, not a proof of model validity."""
    size = len(vector)
    if size < 1 or len(matrix) != size or any(len(row) != size for row in matrix):
        raise ReasoningInputError("A square matrix and matching vector are required")
    if not isfinite(epsilon) or epsilon <= 0:
        raise ReasoningInputError("epsilon must be finite and positive")
    augmented = []
    for row, value in zip(matrix, vector):
        values = [float(item) for item in row] + [float(value)]
        if not all(isfinite(item) for item in values):
            raise ReasoningInputError("Matrix and vector values must be finite")
        augmented.append(values)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= epsilon:
            raise ReasoningInputError("Linear system is singular or numerically ill-conditioned")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [item / divisor for item in augmented[column]]
        for row in range(size):
            if row == column: continue
            factor = augmented[row][column]
            augmented[row] = [left - factor * right
                              for left, right in zip(augmented[row], augmented[column])]
    solution = [augmented[row][-1] for row in range(size)]
    residual = max(abs(sum(float(a) * x for a, x in zip(row, solution)) - float(b))
                   for row, b in zip(matrix, vector))
    return {"solution": solution, "max_absolute_residual": residual,
            "method": "Gaussian elimination with partial pivoting"}


def rank_hypotheses(hypotheses: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute conditional posterior odds from explicitly supplied priors and log Bayes factors."""
    if not hypotheses:
        raise ReasoningInputError("At least one hypothesis is required")
    output = []
    for item in hypotheses:
        identifier = str(item.get("id", "")).strip()
        prior = item.get("prior_probability")
        evidence = item.get("evidence", [])
        if not identifier or isinstance(prior, bool) or not isinstance(prior, (int, float)) or not 0 < prior < 1:
            raise ReasoningInputError("Each hypothesis needs an ID and prior_probability strictly between zero and one")
        if not isinstance(evidence, list):
            raise ReasoningInputError("Hypothesis evidence must be a list")
        evidence_ids = set()
        total_log_bf = 0.0
        for record in evidence:
            evidence_id = str(record.get("id", "")).strip()
            value = record.get("log_bayes_factor")
            if not evidence_id or evidence_id in evidence_ids:
                raise ReasoningInputError("Evidence IDs must be non-empty and unique per hypothesis")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ReasoningInputError("Every evidence record requires a finite log_bayes_factor")
            evidence_ids.add(evidence_id)
            total_log_bf += float(value)
        prior_log_odds = log(float(prior) / (1 - float(prior)))
        log_odds = prior_log_odds + total_log_bf
        probability = 1 / (1 + exp(-max(-700.0, min(700.0, log_odds))))
        output.append({"id": identifier, "conditional_probability": probability,
                       "prior_probability": float(prior), "sum_log_bayes_factors": total_log_bf,
                       "evidence_count": len(evidence)})
    output.sort(key=lambda item: item["conditional_probability"], reverse=True)
    return {"ranked": output,
            "warning": "These conditional values are only as valid as supplied priors and likelihood ratios; they are not empirically calibrated confidence scores."}
