"""LangGraph orchestration for the Merchant Insights Copilot.

Regulatory thresholds are injected as explicit, compliance-team-owned policy
data. Regulatory questions fail closed when no approved policy is configured;
the agent does not guess entity-specific legal limits.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, TypedDict

from modules.copilot import MerchantCopilot, _guardrails_validate, _span

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # Optional runtime dependency
    StateGraph = None
    _LANGGRAPH_IMPORT_ERROR = exc
else:
    _LANGGRAPH_IMPORT_ERROR = None


_REGULATORY_TERMS = re.compile(r"\b(fincen|rbi|pmla|bsa|aml|ctr|sar|str|statutory|reporting threshold)\b", re.I)
_EVASION_ADVICE = re.compile(
    r"\b(split|structure|break up|route|disguise|conceal)\b.{0,90}\b(avoid|evade|bypass|stay below|escape)\b"
    r"|\b(avoid|evade|bypass|stay below|escape)\b.{0,90}\b(reporting|ctr|sar|str|threshold|detection)\b",
    re.I,
)
_THRESHOLD_CLAIM = re.compile(
    r"\b(?:threshold|limit|reportable at|reporting requirement(?: begins)? at)\b"
    r"[^.!?\n]{0,70}?(?P<currency_before>USD|INR|US dollars?|Indian rupees?|rupees?|dollars?|\$|₹|Rs\.?)?\s*"
    r"(?P<value>[\d][\d,]*(?:\.\d+)?)(?:\s*(?P<currency_after>USD|INR|US dollars?|Indian rupees?|rupees?|dollars?|\$|₹|Rs\.?))?",
    re.I,
)


class CopilotState(TypedDict, total=False):
    question: str
    frame: Any
    filters: dict[str, Any]
    compliance_policy: dict[str, Any]
    regulatory_question: bool
    policy_ready: bool
    policy_authorities: list[str]
    policy_context: str
    result: dict[str, Any]


def _load_policy() -> dict[str, Any]:
    raw = os.getenv("COPILOT_COMPLIANCE_POLICY_JSON", "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("COPILOT_COMPLIANCE_POLICY_JSON must be a JSON object")
    return value


def _compliance_failure(answer: str, policy: dict[str, Any], authorities: list[str] | None = None) -> str | None:
    if _EVASION_ADVICE.search(answer):
        return "Response contained advice to evade financial reporting."
    policy_data = policy.get("jurisdictions", policy)
    approved_limits: set[tuple[str | None, float]] = set()
    if isinstance(policy_data, dict):
        allowed_jurisdictions = {name.lower() for name in (authorities or [])}
        for name, jurisdiction in policy_data.items():
            if allowed_jurisdictions and str(name).lower() not in allowed_jurisdictions:
                continue
            if not isinstance(jurisdiction, dict):
                continue
            for rule in jurisdiction.get("rules", []):
                value = rule.get("limit_value") if isinstance(rule, dict) else None
                if value is not None:
                    try:
                        currency = str(rule.get("currency", "")).strip().upper() or None
                        approved_limits.add((currency, float(value)))
                    except (TypeError, ValueError):
                        continue
    for claim in _THRESHOLD_CLAIM.finditer(answer):
        try:
            claimed_limit = float(claim.group("value").replace(",", ""))
        except ValueError:
            continue
        raw_currency = (claim.group("currency_before") or claim.group("currency_after") or "").strip().upper()
        aliases = {
            "$": "USD", "US DOLLAR": "USD", "US DOLLARS": "USD",
            "₹": "INR", "RS": "INR", "RS.": "INR", "RUPEE": "INR",
            "RUPEES": "INR", "INDIAN RUPEE": "INR", "INDIAN RUPEES": "INR",
        }
        claimed_currency = aliases.get(raw_currency, raw_currency or None)
        if not approved_limits or not any(
            abs(claimed_limit - allowed) < 0.005 and claimed_currency is not None and claimed_currency == currency
            for currency, allowed in approved_limits
        ):
            return "Response stated a threshold outside the approved compliance policy."
    return None


def _guardrails_compliance_validate(answer: str, policy: dict[str, Any], authorities: list[str]) -> tuple[bool, str | None]:
    """Run approved-rule checks through Guardrails AI and a fail-closed fallback."""
    failure = _compliance_failure(answer, policy, authorities)
    try:
        from guardrails import Guard
        from guardrails.validator_base import FailResult, PassResult, Validator

        class ApprovedComplianceValidator(Validator):
            def __init__(self):
                super().__init__(on_fail="exception")

            def validate(self, value, metadata):
                message = _compliance_failure(value, metadata.get("policy", {}), metadata.get("authorities", []))
                if message:
                    return FailResult(error_message=message)
                return PassResult()

        result = Guard().use(ApprovedComplianceValidator()).validate(
            answer, metadata={"policy": policy, "authorities": authorities}
        )
        if not getattr(result, "validation_passed", True):
            return False, failure or "Guardrails AI rejected the compliance response."
    except ImportError:
        # The application already uses deterministic fallback checks when the
        # optional Guardrails runtime cannot be imported.
        pass
    except Exception as exc:
        return False, failure or f"Guardrails AI compliance validation failed: {exc}"
    return (not failure), failure


class CopilotAgentGraph:
    """Small LangGraph pipeline around the existing analytics Copilot."""

    def __init__(self, copilot: MerchantCopilot | None = None, compliance_policy: dict[str, Any] | None = None):
        if StateGraph is None:
            raise ImportError("LangGraph runtime is missing; install requirements-advanced.txt") from _LANGGRAPH_IMPORT_ERROR
        self.copilot = copilot or MerchantCopilot()
        self.compliance_policy = compliance_policy if compliance_policy is not None else _load_policy()
        workflow = StateGraph(CopilotState)
        workflow.add_node("prepare", self._traced("prepare", self._prepare))
        workflow.add_node("answer", self._traced("answer", self._answer))
        workflow.add_node("validate", self._traced("validate", self._validate))
        workflow.add_edge(START, "prepare")
        workflow.add_edge("prepare", "answer")
        workflow.add_edge("answer", "validate")
        workflow.add_edge("validate", END)
        self.graph = workflow.compile()

    @staticmethod
    def _traced(name: str, fn):
        def run(state: CopilotState) -> dict[str, Any]:
            with _span(f"copilot.langgraph.node.{name}", **{"langgraph.node": name}):
                return fn(state)
        return run

    def _prepare(self, state: CopilotState) -> dict[str, Any]:
        question = str(state.get("question", "")).strip()
        if not question:
            raise ValueError("question is required")
        regulatory = bool(_REGULATORY_TERMS.search(question))
        policy = state.get("compliance_policy", self.compliance_policy)
        if policy and not isinstance(policy, dict):
            raise ValueError("compliance_policy must be a JSON object")
        policy_data = (policy or {}).get("jurisdictions", policy or {})
        if not isinstance(policy_data, dict):
            policy_data = {}
        requested = [name for name in ("FinCEN", "RBI") if re.search(rf"\b{name}\b", question, re.I)]
        policy_ready = bool(policy_data) and (bool(requested) or not regulatory)
        for authority in requested:
            jurisdiction = next((value for key, value in policy_data.items() if str(key).lower() == authority.lower()), None)
            rules = jurisdiction.get("rules", []) if isinstance(jurisdiction, dict) else []
            if not rules or not all(
                isinstance(rule, dict) and all(rule.get(key) for key in ("rule_id", "applies_to", "trigger", "source_url", "effective_date"))
                and (
                    rule.get("limit_value") is None
                    or (rule.get("currency") and isinstance(rule.get("limit_value"), (int, float)) and rule["limit_value"] >= 0)
                )
                for rule in rules
            ):
                policy_ready = False
                break
        return {
            "question": question,
            "regulatory_question": regulatory,
            "compliance_policy": policy or {},
            "policy_ready": policy_ready,
            "policy_authorities": requested,
            "policy_context": json.dumps(policy, sort_keys=True, ensure_ascii=False) if policy else "",
        }

    def _answer(self, state: CopilotState) -> dict[str, Any]:
        if state.get("regulatory_question") and not state.get("policy_ready"):
            return {"result": {
                "answer": "I can’t provide FinCEN/RBI threshold guidance until an approved compliance policy is configured for the regulated entity and transaction type.",
                "sources": [], "tool_outputs": {}, "chunks_used": [], "chunk_ids": [],
                "state": "compliance_policy_required",
            }}
        question = state["question"]
        if state.get("policy_context"):
            question += "\n\nApproved compliance policy context (treat as binding; do not infer or extend these rules):\n" + state["policy_context"]
        with _span("copilot.langgraph.invoke", **{"copilot.regulatory_question": bool(state.get("regulatory_question"))}):
            result = self.copilot.answer(question, state["frame"], state.get("filters") or {})
        return {"result": result}

    def _validate(self, state: CopilotState) -> dict[str, Any]:
        result = dict(state.get("result") or {})
        answer = str(result.get("answer", ""))
        if state.get("regulatory_question") and state.get("compliance_policy"):
            valid, reason = _guardrails_compliance_validate(
                answer, state["compliance_policy"], state.get("policy_authorities", [])
            )
            if not valid:
                return {"result": self._fallback(result, reason or "Guardrails rejected the compliance response.")}
        if result.get("state") == "answered":
            valid, reason = _guardrails_validate(answer, {"chunk_ids": result.get("chunk_ids", [])})
            if not valid:
                return {"result": self._fallback(result, reason or "Output policy rejected the response.")}
        return {"result": result}

    @staticmethod
    def _fallback(result: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            **result,
            "answer": "The generated answer was withheld because it did not pass the configured compliance and evidence checks.",
            "state": "fallback",
            "fallback_reason": reason,
        }

    def invoke(self, question: str, frame: Any, filters: dict[str, Any] | None = None,
               compliance_policy: dict[str, Any] | None = None) -> dict[str, Any]:
        state: CopilotState = {"question": question, "frame": frame, "filters": filters or {}}
        if compliance_policy is not None:
            state["compliance_policy"] = compliance_policy
        final = self.graph.invoke(state)
        return final["result"]


def build_agent_graph(copilot: MerchantCopilot | None = None,
                      compliance_policy: dict[str, Any] | None = None) -> CopilotAgentGraph:
    """Construct the compiled state graph with an optional approved policy."""
    return CopilotAgentGraph(copilot, compliance_policy)
