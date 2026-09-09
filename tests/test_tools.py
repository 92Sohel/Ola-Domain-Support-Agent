"""
tests/test_tools.py - Validation Tests for Support Ticket Lookup and Escalation Scoring
Track: Business Operations / Customer Support (Ola)

Verifies Part 2 Task 6 & Part 4 Task 15:
- Ticket status lookup
- Escalation score calculation and threshold recommendation
- Least autonomy enforcement
"""

import pytest
from tools import check_support_ticket_status, calculate_escalation_score, ESCALATION_RECOMMENDATION_THRESHOLD
from governance import authorize_tool_invocation, SecurityGovernanceError


def test_ticket_lookup_existing():
    res = check_support_ticket_status("OLA-TCK-1001")
    assert res["found"] is True
    assert res["record_id"] == "OLA-TCK-1001"
    assert res["status"] in ["Open", "In Progress", "Escalated", "Resolved", "Closed"]
    assert 0.0 <= res["escalation_score"] <= 1.0


def test_ticket_lookup_missing():
    res = check_support_ticket_status("OLA-TCK-9999")
    assert res["found"] is False
    assert "error" in res
    assert res["status"] == "Unknown"


def test_escalation_score_bounds_and_recommendation():
    # Un-escalated new ticket
    score_low = calculate_escalation_score(escalated=False, days_since_created=0)
    assert score_low == 0.0
    
    # Escalated old ticket
    score_high = calculate_escalation_score(escalated=True, days_since_created=30)
    assert score_high == 1.0
    assert score_high >= ESCALATION_RECOMMENDATION_THRESHOLD
    
    # Check intermediate
    score_mid = calculate_escalation_score(escalated=True, days_since_created=15)
    assert 0.0 <= score_mid <= 1.0


def test_least_autonomy_enforcement():
    # Authorized agent role
    assert authorize_tool_invocation("Ola Ticket Operations Specialist", "check_support_ticket_status") is True
    
    # Unauthorized agent role
    with pytest.raises(SecurityGovernanceError):
        authorize_tool_invocation("Ola Knowledge Base Retrieval Specialist", "check_support_ticket_status")
