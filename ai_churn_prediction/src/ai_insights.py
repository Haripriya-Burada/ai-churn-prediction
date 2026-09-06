"""
ai_insights.py
----------------
Integrates with Claude (Anthropic) API to generate business insights.
Includes a complete rule-based fallback when the API key is unavailable.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


def get_ai_insights(analytics_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate AI-powered business insights.

    Tries Claude API first; falls back to rule-based engine if the API key
    is missing or the call fails.

    Parameters
    ----------
    analytics_summary : dict
        Summarised, anonymised analytics (churn rate, top drivers, segments, etc.)

    Returns
    -------
    dict with keys: source ("claude" | "fallback"), content (str), success (bool)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    if api_key and api_key != "your_api_key_here":
        result = _call_claude(api_key, analytics_summary)
        if result["success"]:
            return result

    # Fallback
    return _generate_fallback_insights(analytics_summary)


def _call_claude(api_key: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Call Claude API with anonymised analytics summary."""
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        prompt = _build_prompt(summary)

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        content = message.content[0].text
        return {"source": "claude", "content": content, "success": True}

    except Exception as e:
        return {"source": "claude", "content": f"API error: {str(e)}", "success": False}


def _build_prompt(summary: Dict[str, Any]) -> str:
    """Build a structured prompt from the analytics summary."""
    parts = [
        "You are a senior business analyst. Based on the following customer analytics summary, "
        "provide actionable business insights.\n",
        "FORMAT YOUR RESPONSE WITH THESE SECTIONS:",
        "1. **Executive Summary** (2-3 sentences)",
        "2. **Major Churn Drivers** (bullet points)",
        "3. **Customer Segment Insights** (bullet points)",
        "4. **High-Risk Customer Insights** (bullet points)",
        "5. **Business Recommendations** (numbered list)",
        "6. **Retention Strategies** (numbered list)",
        "7. **Suggested KPIs to Monitor** (bullet points)\n",
        "ANALYTICS SUMMARY:",
    ]

    for key, value in summary.items():
        parts.append(f"- {key}: {value}")

    return "\n".join(parts)


def _generate_fallback_insights(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rule-based fallback insight generator.
    Produces structured insights without needing an API.
    """
    churn_rate = summary.get("churn_rate", 0)
    top_drivers = summary.get("top_churn_drivers", [])
    total_customers = summary.get("total_customers", 0)
    high_risk_count = summary.get("high_risk_count", 0)
    avg_tenure = summary.get("avg_tenure", 0)
    avg_monthly = summary.get("avg_monthly_charges", 0)
    segments = summary.get("segments", {})

    lines = []

    # ─── Executive Summary ───
    lines.append("## 📊 Executive Summary\n")
    if churn_rate > 25:
        lines.append(
            f"The current churn rate of **{churn_rate:.1f}%** is above the industry average "
            f"of ~20%, indicating an urgent need for retention interventions. "
            f"Out of **{total_customers:,}** customers, **{high_risk_count:,}** are at high risk "
            f"of leaving."
        )
    elif churn_rate > 15:
        lines.append(
            f"The churn rate of **{churn_rate:.1f}%** is within the typical range but represents "
            f"a significant revenue risk. Targeted interventions for the **{high_risk_count:,}** "
            f"high-risk customers should be prioritised."
        )
    else:
        lines.append(
            f"The churn rate of **{churn_rate:.1f}%** is below industry average, which is positive. "
            f"Focus on maintaining current retention strategies and monitoring the **{high_risk_count:,}** "
            f"high-risk customers."
        )

    # ─── Major Churn Drivers ───
    lines.append("\n## 🔍 Major Churn Drivers\n")
    if top_drivers:
        for i, driver in enumerate(top_drivers[:5], 1):
            feature = driver if isinstance(driver, str) else driver.get("feature", "Unknown")
            lines.append(f"{i}. **{feature}** — significantly impacts customer churn probability.")
    else:
        lines.append("- Churn drivers have not been computed yet. Train a model first.")

    # ─── Customer Segment Insights ───
    lines.append("\n## 👥 Customer Segment Insights\n")
    if segments:
        for seg_name, seg_info in segments.items():
            count = seg_info.get("count", "?")
            lines.append(f"- **{seg_name}**: {count} customers.")
    else:
        lines.append("- Customer segmentation has not been performed yet.")

    # ─── High-Risk Insights ───
    lines.append("\n## ⚠️ High-Risk Customer Insights\n")
    if high_risk_count > 0:
        pct = round(high_risk_count / max(total_customers, 1) * 100, 1)
        lines.append(
            f"- **{high_risk_count:,}** customers ({pct}%) are classified as high risk."
        )
        lines.append("- These customers should receive immediate outreach from the retention team.")
        lines.append("- Common patterns include short tenure, month-to-month contracts, and high support interactions.")
    else:
        lines.append("- No high-risk customers identified or predictions not yet generated.")

    # ─── Business Recommendations ───
    lines.append("\n## 💡 Business Recommendations\n")
    recs = [
        "Launch a contract migration campaign offering 15–20% discounts for switching from monthly to annual plans.",
        f"Implement a proactive support programme for customers with more than 4 support calls per quarter.",
        "Create an early-warning dashboard alerting the retention team when a customer's risk score exceeds 60%.",
        "Develop personalised pricing tiers for high-value customers at risk of churning.",
        "Introduce a customer satisfaction feedback loop within the first 90 days of onboarding.",
    ]
    for i, rec in enumerate(recs, 1):
        lines.append(f"{i}. {rec}")

    # ─── Retention Strategies ───
    lines.append("\n## 🛡️ Retention Strategies\n")
    strategies = [
        "**Win-Back Campaign**: Target recently churned customers with a special offer within 30 days.",
        "**Loyalty Programme**: Reward customers who pass 12, 24, and 36-month milestones.",
        "**Proactive NPS Surveys**: Send quarterly Net Promoter Score surveys and act on detractors immediately.",
        "**Product Education**: Create onboarding email sequences highlighting underutilised features.",
        "**VIP Support**: Offer premium support channels to high-value customers.",
    ]
    for i, s in enumerate(strategies, 1):
        lines.append(f"{i}. {s}")

    # ─── KPIs ───
    lines.append("\n## 📈 Suggested KPIs to Monitor\n")
    kpis = [
        "Monthly Churn Rate (target: < 2%)",
        "Customer Lifetime Value (CLV)",
        "Net Promoter Score (NPS)",
        "Average Revenue Per User (ARPU)",
        "Support Ticket Resolution Time",
        "First-90-Day Retention Rate",
        "Contract Upgrade Rate",
    ]
    for kpi in kpis:
        lines.append(f"- {kpi}")

    content = "\n".join(lines)
    return {"source": "fallback", "content": content, "success": True}


def build_analytics_summary(
    df=None,
    kpis: dict = None,
    importance_df=None,
    risk_df=None,
    segments: dict = None,
) -> Dict[str, Any]:
    """
    Aggregate analytics data into a single summary dict for the AI prompt.
    Only sends anonymised, aggregate-level statistics.
    """
    summary: Dict[str, Any] = {}

    if kpis:
        summary["total_customers"] = kpis.get("total_customers", 0)
        summary["churn_rate"] = kpis.get("churn_rate", 0)
        summary["avg_monthly_charges"] = kpis.get("avg_monthly_charges", 0)
        summary["avg_tenure"] = kpis.get("avg_tenure", 0)

    if importance_df is not None and len(importance_df) > 0:
        summary["top_churn_drivers"] = importance_df.head(5)["Feature"].tolist()

    if risk_df is not None and "RiskLevel" in risk_df.columns:
        summary["high_risk_count"] = int((risk_df["RiskLevel"] == "HIGH").sum())
        summary["medium_risk_count"] = int((risk_df["RiskLevel"] == "MEDIUM").sum())
        summary["low_risk_count"] = int((risk_df["RiskLevel"] == "LOW").sum())

    if segments:
        summary["segments"] = segments

    return summary
