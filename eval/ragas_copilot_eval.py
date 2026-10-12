"""Run the 50-query RAGAS audit against the Copilot and export JSON scores.

Example: python eval/ragas_copilot_eval.py --input path/to/orders.csv
Requires OPENAI_API_KEY. The reference for each case is the deterministic
analytics tool output captured before the LLM answer is scored.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd

QUERY_TEMPLATES = [
    "What does the executive summary say about {segment}?",
    "What discount does the pricing analysis recommend for {segment}?",
    "Are there unusual or fraud related orders in {segment}?",
    "How is customer retention changing for {segment}?",
    "What are the latest revenue and AOV trends for {segment}?",
    "Which customers are at risk and what revenue is exposed in {segment}?",
    "What is the customer lifetime value picture for {segment}?",
    "Are there inventory risks or stockout concerns for {segment}?",
    "How price sensitive is demand in {segment}?",
    "What are the largest growth opportunities and risks for {segment}?",
]
SEGMENTS = ["Electronics", "Fashion", "Home and Kitchen", "Beauty", "Grocery"]
TEST_QUERIES = [template.format(segment=segment) for template in QUERY_TEMPLATES for segment in SEGMENTS]
assert len(TEST_QUERIES) == 50


async def evaluate(rows: list[dict]) -> dict:
    from openai import AsyncOpenAI, OpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness,
    )

    async_llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI())
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=OpenAI(), interface="modern")
    metrics = {
        "context_precision": ContextPrecisionWithReference(llm=async_llm),
        "context_recall": ContextRecall(llm=async_llm),
        "faithfulness": Faithfulness(llm=async_llm),
        "answer_relevancy": AnswerRelevancy(llm=async_llm, embeddings=embeddings),
    }
    audited = []
    for row in rows:
        metric_scores = {}
        for name, metric in metrics.items():
            kwargs = {
                "user_input": row["user_input"],
                "response": row["response"],
                "retrieved_contexts": row["retrieved_contexts"],
            }
            if name in {"context_precision", "context_recall"}:
                kwargs["reference"] = row["reference"]
            result = await metric.ascore(**kwargs)
            metric_scores[name] = float(result.value)
        audited.append({**row, "scores": metric_scores})
    averages = {
        name: sum(item["scores"][name] for item in audited) / len(audited)
        for name in metrics
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(audited),
        "metric_averages": averages,
        "cases": audited,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Order CSV used by the analytics Copilot")
    parser.add_argument("--output", default="eval/ragas_copilot_audit.json")
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required for Copilot generation and RAGAS judging")
    from data.loader import load
    from modules.copilot import MerchantCopilot

    frame = load(args.input, enrich_live=False)
    copilot = MerchantCopilot()
    rows = []
    for query in TEST_QUERIES:
        result = copilot.answer(query, frame)
        reference = "\n\n".join(result.get("tool_outputs", {}).values()).strip()
        rows.append({
            "user_input": query,
            "retrieved_contexts": result.get("chunks_used", []),
            "response": result.get("answer", ""),
            "reference": reference or result.get("answer", ""),
            "chunk_ids": result.get("chunk_ids", []),
            "state": result.get("state", "unknown"),
        })
    report = asyncio.run(evaluate(rows))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {report['dataset_size']} Copilot evaluations to {path}")
    print(json.dumps(report["metric_averages"], indent=2))


if __name__ == "__main__":
    main()
