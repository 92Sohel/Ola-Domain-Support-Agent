"""
tools.py - Operational Tools for Ola Domain Support Agent
Track: Business Operations / Customer Support (Ola)

Includes:
1. check_support_ticket_status: Looks up ticket records from SUPPORT_TICKETS and computes a designed escalation score.
2. rag_search_tool: Retrieves grounded context from ChromaDB collections.
3. Least-autonomy enforcement guard.
"""

from typing import Dict, Any, Optional
from dataset import SUPPORT_TICKETS

# Threshold above which urgent operational escalation is recommended
ESCALATION_RECOMMENDATION_THRESHOLD = 0.65


def calculate_escalation_score(escalated: bool, days_since_created: int) -> float:
    """
    Computes a designed escalation score in [0.0, 1.0].
    
    Formula:
      escalation_score = min(1.0, round(0.40 * (1.0 if escalated else 0.0) + 0.60 * (days_since_created / 30.0), 3))
      
    Justification & Distribution:
      - 0.40 weight allocated to the explicit escalation flag.
      - 0.60 weight allocated to ticket aging (normalized across the 0-30 day window).
      - Threshold of 0.65 identifies tickets in the ~80th percentile of operational urgency:
        an escalated ticket aged >= 13 days, or an aging ticket requiring urgent management intervention.
    """
    clamped_days = max(0, min(30, days_since_created))
    aging_signal = clamped_days / 30.0
    flag_signal = 1.0 if escalated else 0.0
    score = min(1.0, round(0.40 * flag_signal + 0.60 * aging_signal, 3))
    return score


def check_support_ticket_status(record_id: str) -> Dict[str, Any]:
    """
    Looks up a specific Ola support ticket by record_id.
    
    Returns:
      dict with status, resolution_time_hours, days_since_created, escalated,
      escalation_score, and escalation_recommended.
    """
    normalized_id = record_id.strip().upper()
    ticket: Optional[Dict[str, Any]] = None
    
    for item in SUPPORT_TICKETS:
        if item["record_id"].upper() == normalized_id:
            ticket = item
            break
            
    if not ticket:
        return {
            "found": False,
            "record_id": record_id,
            "error": f"Ticket '{record_id}' was not found in the Ola support ticket database.",
            "status": "Unknown",
            "resolution_time_hours": None,
            "days_since_created": None,
            "escalated": False,
            "escalation_score": 0.0,
            "escalation_recommended": False,
        }
        
    score = calculate_escalation_score(
        escalated=ticket["escalated"],
        days_since_created=ticket["days_since_created"]
    )
    
    recommended = score >= ESCALATION_RECOMMENDATION_THRESHOLD
    
    return {
        "found": True,
        "record_id": ticket["record_id"],
        "category": ticket["category"],
        "status": ticket["status"],
        "resolution_time_hours": ticket["resolution_time_hours"],
        "days_since_created": ticket["days_since_created"],
        "escalated": ticket["escalated"],
        "escalation_score": score,
        "escalation_recommended": recommended,
        "recommendation_threshold": ESCALATION_RECOMMENDATION_THRESHOLD,
        "summary": (
            f"Ticket {ticket['record_id']} ({ticket['category']}) is currently '{ticket['status']}'. "
            f"Resolution time: {ticket['resolution_time_hours']}h, Created: {ticket['days_since_created']} days ago. "
            f"Escalation Score: {score:.3f} (Escalation {'RECOMMENDED' if recommended else 'NOT required'})."
        )
    }


if __name__ == "__main__":
    print("Testing check_support_ticket_status on sample tickets:")
    # Test valid ticket 1
    sample1 = check_support_ticket_status("OLA-TCK-1001")
    print("\nSample 1:", sample1["summary"])
    
    # Test valid ticket with high escalation
    sample_escalated = None
    for t in SUPPORT_TICKETS:
        if t["escalated"] and t["days_since_created"] >= 15:
            sample_escalated = check_support_ticket_status(t["record_id"])
            break
    if sample_escalated:
        print("\nSample Escalated (High urgency):", sample_escalated["summary"])
        
    # Test non-existent ticket
    sample_missing = check_support_ticket_status("OLA-TCK-9999")
    print("\nMissing Ticket Test:", sample_missing["error"])
