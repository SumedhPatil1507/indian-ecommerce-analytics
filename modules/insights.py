"""
modules/insights.py  Executive Summary + Recommendations engine.
Generates auto-written business narratives and prioritised action items
directly from the data. No charts  pure business intelligence text.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


#  Executive Summary 

def executive_summary(df: pd.DataFrame, fx: float = 84.0) -> dict:
    """
    Generate a structured executive summary from order-level data.

    Returns a dict with keys:
        headline, kpis, top_insights, risks, opportunities
    """
    rev_total   = df["revenue"].sum()
    rev_cr      = rev_total / 1e7
    rev_usd     = rev_total / fx / 1e6
    orders      = len(df)
    aov         = df["revenue"].mean()
    avg_disc    = df["discount_percent"].mean()
    top_cat     = df.groupby("category")["revenue"].sum().idxmax()
    top_zone    = df.groupby("zone")["revenue"].sum().idxmax()
    top_state   = df.groupby("state")["revenue"].sum().idxmax()
    fest_rev    = df[df["sales_event"] == "Festival"]["revenue"].sum()
    fest_pct    = fest_rev / rev_total * 100 if rev_total else 0
    premium_rev = df[df["brand_type"] == "Premium"]["revenue"].sum()
    premium_pct = premium_rev / rev_total * 100 if rev_total else 0
    high_comp   = df[df["competition_intensity"] == "High"]
    high_comp_disc = high_comp["discount_percent"].mean() if len(high_comp) else avg_disc

    # month-over-month growth
    monthly = df.groupby("year_month")["revenue"].sum().sort_index()
    mom_growth = None
    if len(monthly) >= 2:
        mom_growth = (monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100

    # margin pressure estimate (discount cost)
    disc_cost = (df["base_price"] - df["final_price"]).sum()
    disc_cost_cr = disc_cost / 1e7

    kpis = {
        "Total Revenue":        f"{rev_cr:.1f} Cr  (${rev_usd:.1f}M USD)",
        "Total Orders":         f"{orders:,}",
        "Avg Order Value":      f"{aov:,.0f}",
        "Avg Discount":         f"{avg_disc:.1f}%",
        "Festival Revenue %":   f"{fest_pct:.1f}%",
        "Premium Brand %":      f"{premium_pct:.1f}%",
        "Top Category":         top_cat,
        "Top Zone":             top_zone,
        "Top State":            top_state,
        "MoM Growth":           f"{mom_growth:+.1f}%" if mom_growth is not None else "N/A",
        "Total Discount Cost":  f"{disc_cost_cr:.1f} Cr",
    }

    top_insights = [
        f"**{top_cat}** is the highest-revenue category, driving outsized contribution to total sales.",
        f"**{top_zone} zone** leads all regions  logistics and marketing investment here yields highest ROI.",
        f"Festival periods account for **{fest_pct:.0f}%** of total revenue  seasonal planning is critical.",
        f"Average discount of **{avg_disc:.1f}%** is costing {disc_cost_cr:.1f} Cr in foregone revenue.",
        f"Premium brands represent **{premium_pct:.0f}%** of revenue  premiumisation trend {'is strong' if premium_pct > 40 else 'has room to grow'}.",
    ]
    if mom_growth is not None:
        direction = "growing" if mom_growth > 0 else "declining"
        top_insights.append(f"Revenue is **{direction}** month-over-month at **{mom_growth:+.1f}%**.")

    risks = []
    if avg_disc > 35:
        risks.append(f" Average discount ({avg_disc:.1f}%) is above sustainable threshold  margin erosion risk.")
    if fest_pct > 50:
        risks.append(f" Over {fest_pct:.0f}% revenue concentrated in festival periods  high seasonality risk.")
    if high_comp_disc > avg_disc + 10:
        risks.append(f" High-competition segments show {high_comp_disc:.1f}% avg discount  price war signal.")
    if mom_growth is not None and mom_growth < -5:
        risks.append(f" Revenue declined {mom_growth:.1f}% last month  investigate root cause immediately.")
    if not risks:
        risks.append(" No critical risks detected in current data window.")

    opportunities = _opportunities(df, avg_disc, premium_pct, top_cat, top_zone)

    return {
        "headline": f"{rev_cr:.1f} Cr revenue across {orders:,} orders  {top_cat} & {top_zone} zone lead growth.",
        "kpis": kpis,
        "top_insights": top_insights,
        "risks": risks,
        "opportunities": opportunities,
    }


def _opportunities(df, avg_disc, premium_pct, top_cat, top_zone) -> list[str]:
    opps = []

    # discount reduction opportunity
    if avg_disc > 25:
        recoverable = df["base_price"].sum() * 0.05 / 1e7
        opps.append(
            f" Reducing average discount by 5pp could recover ~{recoverable:.1f} Cr in annual revenue."
        )

    # premium upsell
    if premium_pct < 40:
        opps.append(
            f" Premium brands at {premium_pct:.0f}% of revenue  targeted upsell campaigns could push this to 50%+."
        )

    # zone expansion
    low_zones = df.groupby("zone")["revenue"].sum().nsmallest(2).index.tolist()
    opps.append(
        f" {' & '.join(low_zones)} zones are underperforming  geo-targeted promotions could unlock growth."
    )

    # inventory pressure
    high_pressure = df[df["inventory_pressure"] == "High"]
    if len(high_pressure) / len(df) > 0.4:
        opps.append(
            " 40%+ orders show high inventory pressure  demand forecasting investment would reduce stockouts."
        )

    return opps


#  Recommendations engine 

def generate_recommendations(df: pd.DataFrame) -> list[dict]:
    """
    Generate prioritised, actionable recommendations.
    Each item: {priority, category, action, impact, effort, metric}
    """
    recs = []

    avg_disc = df["discount_percent"].mean()
    monthly  = df.groupby("year_month")["revenue"].sum().sort_index()

    # 1. Discount optimisation
    if avg_disc > 30:
        elastic_cats = _elastic_categories(df)
        for cat in elastic_cats[:2]:
            recs.append({
                "priority": " High",
                "category": "Pricing",
                "action":   f"Reduce discount on {cat} by 58pp during non-festival periods",
                "impact":   f"Est. {df[df['category']==cat]['base_price'].sum()*0.06/1e7:.1f} Cr revenue recovery",
                "effort":   "Low",
                "metric":   "Avg discount %",
            })

    # 2. Festival preparation
    fest_months = df[df["sales_event"] == "Festival"]["month"].value_counts().head(3).index.tolist()
    if fest_months:
        recs.append({
            "priority": " High",
            "category": "Inventory",
            "action":   f"Pre-stock top categories 6 weeks before festival months ({fest_months})",
            "impact":   "Prevent stockouts during peak demand  protect 1525% of annual revenue",
            "effort":   "Medium",
            "metric":   "Inventory pressure %",
        })

    # 3. Zone expansion
    zone_rev = df.groupby("zone")["revenue"].sum()
    bottom_zone = zone_rev.idxmin()
    top_zone    = zone_rev.idxmax()
    recs.append({
        "priority": " Medium",
        "category": "Growth",
        "action":   f"Launch geo-targeted campaign in {bottom_zone} zone (currently {zone_rev[bottom_zone]/zone_rev[top_zone]*100:.0f}% of {top_zone})",
        "impact":   "10% lift in bottom zone = meaningful incremental revenue",
        "effort":   "Medium",
        "metric":   "Zone revenue share",
    })

    # 4. Premium upsell
    premium_pct = (df["brand_type"] == "Premium").mean() * 100
    if premium_pct < 45:
        recs.append({
            "priority": " Medium",
            "category": "Revenue Mix",
            "action":   "Bundle premium products with mass bestsellers  increase premium attach rate",
            "impact":   f"Premium at {premium_pct:.0f}%  target 50% = higher AOV and margins",
            "effort":   "Low",
            "metric":   "Premium revenue %",
        })

    # 5. Retention
    recs.append({
        "priority": " Low",
        "category": "Retention",
        "action":   "Launch loyalty programme targeting top 20% customers by CLV",
        "impact":   "5% retention improvement = 2595% profit increase (Bain & Co benchmark)",
        "effort":   "High",
        "metric":   "Repeat purchase rate",
    })

    # 6. Competition response
    high_comp = df[df["competition_intensity"] == "High"]
    if len(high_comp) / len(df) > 0.3:
        recs.append({
            "priority": " Medium",
            "category": "Competitive",
            "action":   "Shift high-competition SKUs toward value-added bundles rather than pure discounting",
            "impact":   "Protect margins while maintaining conversion in competitive segments",
            "effort":   "Medium",
            "metric":   "Discount % in high-competition segments",
        })

    return sorted(recs, key=lambda r: [" High"," Medium"," Low"].index(r["priority"]))


def _elastic_categories(df: pd.DataFrame) -> list[str]:
    """Return categories where higher discount correlates with higher units (elastic)."""
    results = []
    for cat, grp in df.groupby("category"):
        if len(grp) < 30:
            continue
        corr = grp["discount_percent"].corr(grp["units_sold"])
        if corr > 0.1:
            results.append((cat, corr))
    return [c for c, _ in sorted(results, key=lambda x: -x[1])]
