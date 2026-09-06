"""
recommendations.py
--------------------
Rule-based business recommendation engine.
Generates actionable retention strategies based on customer attributes
and churn risk.
"""

from typing import List, Dict, Any


def generate_recommendations(
    risk_level: str,
    customer_data: dict,
    top_factors: List[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Generate business recommendations based on a customer's risk profile
    and attributes.

    Each recommendation is a dict with 'title', 'description', 'priority'.

    Parameters
    ----------
    risk_level : str   — "LOW", "MEDIUM", or "HIGH"
    customer_data : dict — the customer's features
    top_factors : list  — top contributing factors from the model

    Returns
    -------
    List of recommendation dicts.
    """
    recs: List[Dict[str, str]] = []

    # --- HIGH risk + Short tenure ---
    tenure = customer_data.get("Tenure", None)
    if risk_level == "HIGH" and tenure is not None and tenure < 12:
        recs.append({
            "title": "🎯 Onboarding Support Programme",
            "description": (
                "This customer is new and at high churn risk. Launch a "
                "dedicated onboarding programme with a personal success "
                "manager, guided tutorials, and a welcome discount."
            ),
            "priority": "Critical",
        })

    # --- HIGH risk + High monthly charges ---
    monthly = customer_data.get("MonthlyCharges", None)
    if risk_level in ("HIGH", "MEDIUM") and monthly is not None and monthly > 70:
        recs.append({
            "title": "💰 Customised Pricing Plan",
            "description": (
                "The customer is paying above-average monthly charges. "
                "Offer a tailored pricing plan, annual discount, or "
                "bundle deal to improve perceived value."
            ),
            "priority": "High",
        })

    # --- Month-to-month contract ---
    contract = customer_data.get("Contract", "")
    if "month" in str(contract).lower():
        recs.append({
            "title": "📋 Annual Contract Incentive",
            "description": (
                "Month-to-month customers churn at much higher rates. "
                "Offer a 15-20 % discount for switching to an annual or "
                "two-year contract."
            ),
            "priority": "High",
        })

    # --- High support calls ---
    support = customer_data.get("SupportCalls", None)
    if support is not None and support >= 4:
        recs.append({
            "title": "🛠️ Proactive Support Intervention",
            "description": (
                "This customer contacts support frequently, indicating "
                "possible product or service issues. Assign a dedicated "
                "support agent and conduct a root-cause analysis."
            ),
            "priority": "High",
        })

    # --- Low satisfaction ---
    satisfaction = customer_data.get("SatisfactionScore", None)
    if satisfaction is not None and satisfaction <= 2:
        recs.append({
            "title": "😊 Customer Satisfaction Recovery",
            "description": (
                "Low satisfaction score detected. Schedule a personal "
                "call, send a satisfaction survey to understand pain "
                "points, and offer a goodwill credit or upgrade."
            ),
            "priority": "Critical",
        })

    # --- Low engagement ---
    usage = customer_data.get("UsageFrequency", None)
    if usage is not None and usage < 8:
        recs.append({
            "title": "📢 Re-engagement Campaign",
            "description": (
                "The customer uses the service infrequently. Launch a "
                "personalised email/notification campaign showcasing "
                "features they haven't tried. Consider free premium "
                "trial access."
            ),
            "priority": "Medium",
        })

    # --- Electronic check payment ---
    payment = customer_data.get("PaymentMethod", "")
    if "electronic check" in str(payment).lower():
        recs.append({
            "title": "💳 Payment Method Upgrade",
            "description": (
                "Electronic check users show higher churn. Offer "
                "incentives (small discount or reward points) for "
                "switching to automatic bank transfer or credit card."
            ),
            "priority": "Medium",
        })

    # --- Fiber optic internet ---
    internet = customer_data.get("InternetService", "")
    if "fiber" in str(internet).lower() and risk_level in ("HIGH", "MEDIUM"):
        recs.append({
            "title": "🌐 Fiber Service Quality Review",
            "description": (
                "Fiber optic customers sometimes churn due to service "
                "quality expectations. Ensure SLA compliance, offer a "
                "speed upgrade, or provide a loyalty bonus."
            ),
            "priority": "Medium",
        })

    # --- General LOW risk recommendation ---
    if risk_level == "LOW":
        recs.append({
            "title": "⭐ Loyalty Reward",
            "description": (
                "This customer has low churn risk. Maintain engagement "
                "with a loyalty programme, exclusive offers, or a "
                "referral bonus to turn them into a brand advocate."
            ),
            "priority": "Low",
        })

    # Ensure at least one recommendation
    if not recs:
        recs.append({
            "title": "📊 Standard Monitoring",
            "description": (
                "No specific risk triggers identified. Continue "
                "standard monitoring and periodic check-ins."
            ),
            "priority": "Low",
        })

    return recs


def generate_segment_recommendations(segment_label: str, cluster_stats: dict) -> List[str]:
    """
    Generate high-level recommendations for a customer segment.
    """
    recs = []
    label_lower = segment_label.lower()

    if "high value" in label_lower:
        recs.append("Offer premium loyalty benefits to retain these valuable customers.")
        recs.append("Create VIP support channels and exclusive product access.")

    if "at risk" in label_lower or "risk" in label_lower:
        recs.append("Prioritise proactive outreach for this at-risk segment.")
        recs.append("Investigate common complaints and service issues.")

    if "price sensitive" in label_lower:
        recs.append("Consider value-tier pricing and budget-friendly bundles.")
        recs.append("Highlight cost-per-value in marketing communications.")

    if "low engagement" in label_lower:
        recs.append("Launch targeted re-engagement campaigns with feature education.")
        recs.append("Offer free trials of premium features to demonstrate value.")

    if "loyal" in label_lower:
        recs.append("Introduce a referral programme to leverage loyal advocates.")
        recs.append("Offer anniversary rewards and long-tenure discounts.")

    if not recs:
        recs.append("Monitor this segment's behaviour and personalise communications.")
        recs.append("Gather feedback to understand unique needs.")

    return recs
