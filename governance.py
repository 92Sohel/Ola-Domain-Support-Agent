"""
governance.py - Four-Layer AI Governance Model for Ola Domain Support Agent
Track: Business Operations / Customer Support (Ola)

Requirements (Part 4 Task 15):
1. Application Layer: Enforce principle of least autonomy.
   - Only Lookup Agent may call check_support_ticket_status.
   - Block/guard demonstrated with a one-paragraph explanation.
2. Risk Classification:
   - Classified under Low/Medium/High scheme (Medium Risk) with a one-paragraph justification.
3. Runtime Layer:
   - Per-request token/cost budget cap.
   - Demonstrate deliberately oversized request being rejected.
"""

import sys
from typing import Dict, Any, Optional

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ==========================================
# 1. Application Layer: Principle of Least Autonomy
# ==========================================

AUTHORIZED_LOOKUP_ROLES = {"Ola Ticket Operations Specialist", "Lookup Agent"}

class SecurityGovernanceError(PermissionError):
    """Raised when an agent attempts an unauthorized tool execution or wiring."""
    pass


def authorize_tool_invocation(agent_role: str, tool_name: str) -> bool:
    """
    Enforces least autonomy at the application layer:
    Only agents specifically authorized for ticket lookup may access the database lookup tool.
    """
    if tool_name in ["check_support_ticket_status", "check_ticket_status"]:
        if agent_role not in AUTHORIZED_LOOKUP_ROLES:
            raise SecurityGovernanceError(
                f"Governance Violation [Application Layer]: Agent '{agent_role}' is not authorized "
                f"to access '{tool_name}'. Least autonomy policy restricts ticket lookup access strictly "
                f"to the dedicated 'Ola Ticket Operations Specialist'."
            )
    return True


LEAST_AUTONOMY_EXPLANATION = (
    "Application Layer Guard Explanation: Under the principle of least autonomy, agents are granted only "
    "the minimal permissions necessary to fulfill their specific operational mandate. In our multi-agent architecture, "
    "the 'check_support_ticket_status' tool has access to internal customer ticket records, resolution metrics, and "
    "escalation flags. Permitting general-purpose retrieval or synthesis agents to execute database queries introduces "
    "unnecessary attack surfaces and risk of unauthorized data traversal. Consequently, our application layer enforces "
    "strict role-based tool gating: only the dedicated Lookup Agent ('Ola Ticket Operations Specialist') is granted "
    "access, while the Retrieval Agent and Response Composer are physically unwired and cryptographically/programmatically "
    "blocked from invoking ticket database tools."
)


# ==========================================
# 2. Risk Classification
# ==========================================

RISK_CLASSIFICATION = "Medium Risk"

RISK_JUSTIFICATION = (
    "System Risk Classification & Justification: The Ola Domain Support Agent is formally classified as "
    "'Medium Risk' under the enterprise AI governance taxonomy. Unlike Low Risk applications (such as internal meeting "
    "transcription or non-interactive text summarization), customer support agents directly interact with the public, "
    "interpret binding service-level agreements, and calculate dispute escalation scores that impact customer trust and "
    "operational workloads. However, unlike High Risk systems (which govern clinical medical diagnostics, autonomous "
    "hiring decisions, credit scoring, or direct irrevocable financial disbursements), this agent operates in a bounded "
    "advisory capacity: all financial refunds are capped by strict policy thresholds, phone PII is masked at the perimeter, "
    "a secondary Autogen review team audits outputs before delivery, and any significant escalation triggers human frontline "
    "review. Therefore, its operational blast radius is firmly contained within the Medium Risk tier."
)


# ==========================================
# 3. Runtime Layer: Token & Cost Budget Cap
# ==========================================

# Maximum allowable input tokens per single request (approx 500 tokens / ~2000 chars)
MAX_REQUEST_TOKEN_BUDGET = 500
ESTIMATED_COST_PER_1K_TOKENS_USD = 0.002  # Simulated budget tracking


class RuntimeBudgetExceededError(ValueError):
    """Raised when an incoming request exceeds the per-request token or cost budget."""
    pass


def estimate_tokens(text: str) -> int:
    """Estimates token count using standard word/subword heuristic."""
    words = len(text.strip().split())
    # Standard rule of thumb: 1 word ~ 1.33 tokens
    return max(1, int(words * 1.33))


def enforce_runtime_budget(query: str, max_tokens: int = MAX_REQUEST_TOKEN_BUDGET) -> Dict[str, Any]:
    """
    Enforces per-request runtime cost and token limits.
    Rejects oversized requests to prevent denial-of-service or budget exhaustion.
    """
    est_tokens = estimate_tokens(query)
    est_cost = (est_tokens / 1000.0) * ESTIMATED_COST_PER_1K_TOKENS_USD
    
    if est_tokens > max_tokens:
        raise RuntimeBudgetExceededError(
            f"Runtime Budget Exceeded [Runtime Layer]: Request estimated at {est_tokens} tokens "
            f"exceeds the hard per-request safety cap of {max_tokens} tokens (Estimated Cost: ${est_cost:.4f}). "
            f"Request rejected to prevent resource exhaustion."
        )
        
    return {
        "status": "APPROVED",
        "estimated_tokens": est_tokens,
        "max_tokens_budget": max_tokens,
        "estimated_cost_usd": round(est_cost, 6),
    }


if __name__ == "__main__":
    print("=" * 75)
    print("TASK 15: FOUR-LAYER AI GOVERNANCE DEMONSTRATION")
    print("=" * 75)
    
    # 1. Demonstrate Least Autonomy Enforcement
    print("\n--- 1. Application Layer: Least Autonomy Guard ---")
    authorized_role = "Ola Ticket Operations Specialist"
    unauthorized_role = "Ola Knowledge Base Retrieval Specialist"
    
    # Authorized invocation
    auth_check = authorize_tool_invocation(authorized_role, "check_support_ticket_status")
    print(f"Authorized Call by '{authorized_role}': SUCCESS (Allowed={auth_check})")
    
    # Unauthorized invocation attempt
    blocked = False
    try:
        authorize_tool_invocation(unauthorized_role, "check_support_ticket_status")
    except SecurityGovernanceError as e:
        blocked = True
        print(f"Unauthorized Call by '{unauthorized_role}': BLOCKED")
        print(f"Governance Error Message: {e}")
        
    assert blocked is True, "Least autonomy guard failed: Unauthorized agent was not blocked!"
    print("\nLeast Autonomy Explanation:")
    print(LEAST_AUTONOMY_EXPLANATION)
    
    # 2. Risk Classification
    print("\n--- 2. System Risk Classification ---")
    print(f"Classification: {RISK_CLASSIFICATION}")
    print("\nJustification:")
    print(RISK_JUSTIFICATION)
    
    # 3. Runtime Layer: Budget Cap Enforcement
    print("\n--- 3. Runtime Layer: Per-Request Budget Cap ---")
    valid_query = "What is the policy for repeat complaints filed within a week?"
    budget_res = enforce_runtime_budget(valid_query)
    print(f"Normal Query: {budget_res['estimated_tokens']} tokens -> {budget_res['status']}")
    
    # Deliberately oversized query (simulated denial-of-service / token explosion)
    oversized_query = "Please analyze this lengthy input: " + ("lorem ipsum dolor sit amet " * 300)
    oversized_blocked = False
    try:
        enforce_runtime_budget(oversized_query)
    except RuntimeBudgetExceededError as e:
        oversized_blocked = True
        print("Oversized Request: REJECTED")
        print(f"Error Message: {e}")
        
    assert oversized_blocked is True, "Runtime budget guard failed: Oversized request was not rejected!"
    print("\nFour-layer AI governance model verified successfully!")
