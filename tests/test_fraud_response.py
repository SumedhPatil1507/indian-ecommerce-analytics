"""Tests for the stable fraud assessment response schema and policy."""
from core.fraud_response import FraudAssessment, build_fraud_assessment


def test_response_uses_exact_four_field_schema():
    response = build_fraud_assessment(
        collusion_probability=0.91,
        rto_probability=0.12,
        graph_centrality_score=0.74,
        shap_contributions={"shared_device_count": 0.31, "transaction_velocity": -0.08},
    )
    payload = response.model_dump()
    assert set(payload) == {"verdict", "reasoning", "recommendation", "next_steps"}
    assert isinstance(response, FraudAssessment)
    assert response.verdict == "HIGH_RISK_COLLUSION_RING_DETECTED"
    assert "R-GCN graph centrality score=0.7400" in response.reasoning
    assert "shared_device_count=+0.3100" in response.reasoning
    assert "HITL review" in response.recommendation
    assert len(response.next_steps) == 3


def test_no_assessment_data_fails_safe_without_claiming_clear():
    response = build_fraud_assessment()
    assert response.verdict == "FRAUD_ASSESSMENT_UNAVAILABLE"
    assert "do not auto-approve" in response.recommendation
    assert "unavailable" in response.reasoning


def test_low_risk_result_is_clear_and_shap_values_are_sanitized():
    response = build_fraud_assessment(
        collusion_probability=0.2,
        rto_probability=0.3,
        graph_centrality_score=float("nan"),
        shap_contributions={"bad": float("inf")},
    )
    assert response.verdict == "CLEAR_NO_HIGH_RISK_SIGNAL"
    assert "centrality score=unavailable" in response.reasoning
    assert "contributions=unavailable" in response.reasoning
