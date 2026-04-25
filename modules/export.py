"""
modules/export.py — PDF and Excel export for client deliverables.
"""
from __future__ import annotations
import io
from datetime import datetime

import pandas as pd


# ── Excel export ──────────────────────────────────────────────────────────────

def to_excel(df: pd.DataFrame, summary: dict) -> bytes:
    """
    Export dataset + executive summary to a formatted Excel workbook.
    Returns bytes ready for st.download_button.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: Executive Summary KPIs
        kpi_df = pd.DataFrame(
            list(summary["kpis"].items()), columns=["Metric", "Value"]
        )
        kpi_df.to_excel(writer, sheet_name="Executive Summary", index=False)

        # Sheet 2: Insights
        ins_df = pd.DataFrame({
            "Top Insights":    summary["top_insights"],
        })
        ins_df.to_excel(writer, sheet_name="Insights", index=False)

        # Sheet 3: Recommendations
        from modules.insights import generate_recommendations
        recs = generate_recommendations(df)
        rec_df = pd.DataFrame(recs)
        rec_df.to_excel(writer, sheet_name="Recommendations", index=False)

        # Sheet 4: Raw data (capped at 50k rows for file size)
        df.head(50_000).to_excel(writer, sheet_name="Data", index=False)

        # Sheet 5: Monthly revenue
        monthly = df.groupby("year_month")["revenue"].agg(
            total_revenue="sum", avg_order_value="mean", order_count="count"
        ).reset_index()
        monthly.to_excel(writer, sheet_name="Monthly Revenue", index=False)

    return buf.getvalue()


# ── PDF export (via fpdf2) ────────────────────────────────────────────────────

def to_pdf(summary: dict, recs: list[dict]) -> bytes:
    """
    Generate a clean PDF executive report.
    Returns bytes ready for st.download_button.
    Falls back to a plain-text PDF if fpdf2 is not installed.
    """
    try:
        from fpdf import FPDF  # type: ignore
    except ImportError:
        return _plain_text_pdf(summary, recs)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, "IndiaCommerce Analytics", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Executive Report  |  Generated {datetime.now().strftime('%d %b %Y, %H:%M')}", ln=True)
    pdf.ln(4)

    # Headline
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 8, summary.get("headline", ""))
    pdf.ln(4)

    # KPIs
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(245, 247, 250)
    pdf.cell(0, 8, "Key Performance Indicators", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    for k, v in summary.get("kpis", {}).items():
        pdf.cell(70, 7, str(k), border="B")
        pdf.cell(0,  7, str(v), border="B", ln=True)
    pdf.ln(4)

    # Insights
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(245, 247, 250)
    pdf.cell(0, 8, "Top Insights", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    for ins in summary.get("top_insights", []):
        clean = ins.replace("**", "")
        pdf.multi_cell(0, 6, f"• {clean}")
    pdf.ln(4)

    # Risks
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Risks", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    for r in summary.get("risks", []):
        clean = r.replace("⚠️", "!").replace("✅", "OK")
        pdf.multi_cell(0, 6, f"• {clean}")
    pdf.ln(4)

    # Recommendations
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Prioritised Recommendations", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for i, rec in enumerate(recs, 1):
        pdf.set_font("Helvetica", "B", 10)
        label = rec["priority"].replace("🔴","[HIGH]").replace("🟠","[MED]").replace("🟡","[LOW]")
        pdf.cell(0, 7, f"{i}. {label}  |  {rec['category']}", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, f"   Action: {rec['action']}")
        pdf.multi_cell(0, 6, f"   Impact: {rec['impact']}")
        pdf.cell(0, 6, f"   Effort: {rec['effort']}  |  Metric: {rec['metric']}", ln=True)
        pdf.ln(2)

    return bytes(pdf.output())


def _plain_text_pdf(summary: dict, recs: list[dict]) -> bytes:
    """Minimal fallback when fpdf2 is not installed."""
    lines = [
        "INDIACOMMERCE ANALYTICS — EXECUTIVE REPORT",
        f"Generated: {datetime.now().strftime('%d %b %Y')}",
        "",
        summary.get("headline", ""),
        "",
        "KPIs",
    ]
    for k, v in summary.get("kpis", {}).items():
        lines.append(f"  {k}: {v}")
    lines += ["", "INSIGHTS"]
    for ins in summary.get("top_insights", []):
        lines.append(f"  - {ins.replace('**','')}")
    lines += ["", "RECOMMENDATIONS"]
    for rec in recs:
        lines.append(f"  [{rec['priority']}] {rec['action']}")
    return "\n".join(lines).encode("utf-8")
