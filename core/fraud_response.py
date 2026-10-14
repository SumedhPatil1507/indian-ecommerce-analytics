"""Canonical JSON contract for fraud assessment endpoints."""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict


class FraudAssessment(BaseModel):
    """The stable four-field fraud assessment response contract."""

    model_config = ConfigDict(extra="forbid")

    verdict: str
    reasoning: str
    recommendation: str
    next_steps: list[str]


def _score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def build_fraud_assessment(
    *,
    collusion_probability: Any = None,
    rto_probability: Any = None,
    graph_centrality_score: Any = None,
    shap_contributions: dict[str, Any] | None = None,
) -> FraudAssessment:
    """Build an explainable, risk-rated result without inventing explanations."""
    collusion = _score(collusion_probability)
    rto = _score(rto_probability)
    centrality = _score(graph_centrality_score)
    contributions = []
    for feature, raw_value in (shap_contributions or {}).items():
        value = _score(raw_value)
        if value is not None:
            contributions.append((str(feature), value))
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)

    assessed = collusion is not None or rto is not None
    if not assessed:
        verdict = "FRAUD_ASSESSMENT_UNAVAILABLE"
        recommendation = "Escalate for manual review; do not auto-approve or freeze payout without an assessment."
    elif collusion is not None and collusion >= 0.8:
        verdict = "HIGH_RISK_COLLUSION_RING_DETECTED"
        recommendation = "Escalate to HITL review; consider a temporary payout hold only after human review."
    elif rto is not None and rto >= 0.8:
        verdict = "HIGH_RISK_RTO_FRAUD_DETECTED"
        recommendation = "Escalate to HITL review and verify delivery and return evidence."
    elif max(value for value in (collusion, rto) if value is not None) >= 0.5:
        verdict = "REVIEW_REQUIRED_ELEVATED_FRAUD_RISK"
        recommendation = "Enqueue for HITL review before taking account or payout action."
    else:
        verdict = "CLEAR_NO_HIGH_RISK_SIGNAL"
        recommendation = "Auto-approve within standard controls; continue routine monitoring."

    centrality_text = f"{centrality:.4f}" if centrality is not None else "unavailable"
    shap_text = ", ".join(f"{name}={value:+.4f}" for name, value in contributions[:5]) or "unavailable"
    score_text = (
        f"R-GCN collusion probability={collusion:.4f}" if collusion is not None else "R-GCN collusion probability unavailable"
    ) + "; " + (
        f"R-GCN RTO probability={rto:.4f}" if rto is not None else "R-GCN RTO probability unavailable"
    )
    reasoning = (
        f"{score_text}; R-GCN graph centrality score={centrality_text}; "
        f"XGBoost SHAP feature contributions={shap_text}."
    )
    return FraudAssessment(
        verdict=verdict,
        reasoning=reasoning,
        recommendation=recommendation,
        next_steps=[
            "Step 1: Enqueue to HITL review queue",
            "Step 2: Cross-reference PAN/SSN in fraud database",
            "Step 3: Dispatch field audit",
        ],
    )
