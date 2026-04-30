"""
modules/export.py - PDF and Excel export for client deliverables.
"""
from __future__ import annotations
import io
import textwrap
from datetime import datetime
import pandas as pd


def _safe(text: str) -> str:
    """Strip non-latin-1 characters so fpdf2 built-in fonts never crash."""
    replacements = {
        "Rs": "Rs", "->": "->",
        "Rs": "Rs", "->": "->", "<-": "<-",
        "Rs": "Rs",
    }
    table = {
        ord("Rs"[0]): "Rs",
        0x20B9: "Rs",   # rupee sign
        0x2192: "->",   # right arrow
        0x2190: "<-",   # left arrow
        0x2022: "-",    # bullet
        0x26A0: "[!]",  # warning
        0x2705: "[OK]", # check mark
        0x1F4A1: "[*]", # bulb
        0x1F534: "[HIGH]",
        0x1F7E0: "[MED]",
        0x1F7E1: "[LOW]",
        0x1F7E2: "[OK]",
        0x1F535: "[INFO]",
    }
    text = text.translate(table)
    text = text.replace("**", "").replace("Rs", "Rs")
    # Remove any remaining non-latin-1
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def _wrap(text: str, width: int = 90) -> list[str]:
    """Wrap long text into lines of max `width` chars."""
    return textwrap.wrap(_safe(text), width=width) or [""]


def to_excel(df: pd.DataFrame, summary: dict) -> bytes:
    """Export dataset + executive summary to Excel. Returns bytes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(list(summary["kpis"].items()), columns=["Metric", "Value"]).to_excel(
            writer, sheet_name="Executive Summary", index=False)
        pd.DataFrame({"Top Insights": summary["top_insights"]}).to_excel(
            writer, sheet_name="Insights", index=False)
        from modules.insights import generate_recommendations
        pd.DataFrame(generate_recommendations(df)).to_excel(
            writer, sheet_name="Recommendations", index=False)
        df.head(50_000).to_excel(writer, sheet_name="Data", index=False)
        df.groupby("year_month")["revenue"].agg(
            total_revenue="sum", avg_order_value="mean", order_count="count"
        ).reset_index().to_excel(writer, sheet_name="Monthly Revenue", index=False)
    return buf.getvalue()


def to_pdf(summary: dict, recs: list[dict]) -> bytes:
    """
    Generate PDF executive report.
    Falls back to plain-text bytes if fpdf2 fails for any reason.
    """
    try:
        from fpdf import FPDF
        return _build_pdf(summary, recs)
    except Exception:
        return _plain_text_pdf(summary, recs)


def _build_pdf(summary: dict, recs: list[dict]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Usable width = page width - left margin - right margin
    W = pdf.w - pdf.l_margin - pdf.r_margin  # ~170mm for A4

    def h1(txt: str) -> None:
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(W, 10, _safe(txt)[:80], ln=True)
        pdf.ln(2)

    def h2(txt: str) -> None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(240, 242, 248)
        pdf.cell(W, 8, _safe(txt)[:80], ln=True, fill=True)
        pdf.ln(1)

    def body(txt: str) -> None:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        for line in _wrap(txt, width=95):
            pdf.cell(W, 5, line, ln=True)

    def kv(key: str, val: str) -> None:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(60, 6, _safe(str(key))[:30], border="B")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(W - 60, 6, _safe(str(val))[:60], border="B", ln=True)

    # ── Page 1 ────────────────────────────────────────────────────────────────
    h1("IndiaCommerce Analytics")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(W, 5, f"Executive Report  |  {datetime.now().strftime('%d %b %Y, %H:%M')}", ln=True)
    pdf.ln(4)

    body(summary.get("headline", ""))
    pdf.ln(4)

    h2("Key Performance Indicators")
    for k, v in summary.get("kpis", {}).items():
        kv(k, v)
    pdf.ln(4)

    h2("Top Insights")
    for ins in summary.get("top_insights", []):
        body(f"- {ins}")
    pdf.ln(4)

    h2("Risks")
    for r in summary.get("risks", []):
        body(f"- {r}")
    pdf.ln(4)

    h2("Opportunities")
    for o in summary.get("opportunities", []):
        body(f"- {o}")

    # ── Page 2: Recommendations ───────────────────────────────────────────────
    pdf.add_page()
    h1("Prioritised Recommendations")
    pdf.ln(2)

    for i, rec in enumerate(recs, 1):
        priority_label = _safe(rec["priority"])
        cat_label      = _safe(rec["category"])
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(W, 7, f"{i}. {priority_label}  |  {cat_label}", ln=True)
        body(f"Action: {rec['action']}")
        body(f"Impact: {rec['impact']}")
        body(f"Effort: {rec['effort']}  |  Metric: {rec['metric']}")
        pdf.ln(3)

    return bytes(pdf.output())


def _plain_text_pdf(summary: dict, recs: list[dict]) -> bytes:
    """Plain-text fallback — always works, no dependencies."""
    lines = [
        "INDIACOMMERCE ANALYTICS - EXECUTIVE REPORT",
        f"Generated: {datetime.now().strftime('%d %b %Y')}",
        "", summary.get("headline", ""), "",
        "=== KPIs ===",
    ]
    for k, v in summary.get("kpis", {}).items():
        lines.append(f"  {k}: {v}")
    lines += ["", "=== INSIGHTS ==="]
    for ins in summary.get("top_insights", []):
        lines.append(f"  - {ins.replace('**','')}")
    lines += ["", "=== RISKS ==="]
    for r in summary.get("risks", []):
        lines.append(f"  - {r}")
    lines += ["", "=== RECOMMENDATIONS ==="]
    for rec in recs:
        lines.append(f"  [{rec['priority']}] {rec['action']}")
        lines.append(f"    Impact: {rec['impact']}")
    return "\n".join(lines).encode("utf-8")
