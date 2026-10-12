"""Copilot output policy and fallback state tests."""
import sys
import types

import pandas as pd

from modules import copilot


def test_unverified_price_adjustment_transitions_to_fallback(monkeypatch):
    chunk_id = "trustedchunk123"

    class Completions:
        def create(self, **kwargs):
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=f"Offer a 40% discount. [chunk:{chunk_id}]")
            )])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=Completions())

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    fake_openai.NotFoundError = type("NotFoundError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    instance = copilot.MerchantCopilot()
    instance._plan_tools = lambda question: ["price_optimizer"]
    instance.run_price_optimizer = lambda df, params=None: "Verified optimizer has no recommendations."
    instance._store("Pricing analysis reports stable margins and demand.", "price_optimizer")
    instance._retrieve = lambda question, top_k=3: [
        {"chunk_id": chunk_id, "source": "price_optimizer", "chunk": "Pricing analysis reports stable margins and demand."}
    ]

    result = instance.answer("Should I increase discounts?", pd.DataFrame({"revenue": [100]}))
    assert result["state"] == "fallback"
    assert "25% cap" in result["fallback_reason"]
    assert "40% discount" not in result["answer"]


def test_price_adjustment_under_cap_still_requires_lerner_verification():
    passed, reason = copilot._guardrails_validate(
        "I recommend a 20% discount. [chunk:trustedchunk123]",
        {"chunk_ids": ["trustedchunk123"], "verified_optimal_discounts": []},
    )
    assert not passed
    assert "Lerner-index" in reason


def test_chunk_citation_must_match_retrieved_chunk():
    passed, reason = copilot._guardrails_validate(
        "Revenue improved. [chunk:madeup]",
        {"chunk_ids": ["trustedchunk123"], "verified_optimal_discounts": []},
    )
    assert not passed
    assert "chunk ID" in reason
