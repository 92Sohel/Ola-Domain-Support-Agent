"""
dataset.py - Deterministic Synthetic Support Ticket Dataset for Ola Domain Support Agent
Track: Business Operations / Customer Support (Ola)

Requirements:
- >=40 records (generates 50 records)
- Category vocabulary: Billing, Technical Issue, Account Access, Product Defect, General Inquiry (every value >=3 records)
- Status vocabulary: Open, In Progress, Escalated, Resolved, Closed (every value >=1 record)
- Realistic resolution_time_hours (1.0 to 96.0 hours)
- days_since_created: integer 0-30
- escalated: boolean (percentage must land strictly between 10% and 30%)
- Seeded, deterministic reproducibility
"""

import random
from typing import Dict, List, Any

RANDOM_SEED = 42

CATEGORIES = [
    "Billing",
    "Technical Issue",
    "Account Access",
    "Product Defect",
    "General Inquiry",
]

STATUSES = [
    "Open",
    "In Progress",
    "Escalated",
    "Resolved",
    "Closed",
]


def generate_support_tickets(seed: int = RANDOM_SEED, total_records: int = 50) -> List[Dict[str, Any]]:
    """
    Generates a deterministic, seeded support ticket dataset for Ola operations.
    """
    rng = random.Random(seed)
    
    category_choices = [
        "Billing",
        "Technical Issue",
        "Account Access",
        "Product Defect",
        "General Inquiry",
    ]
    category_weights = [0.28, 0.24, 0.18, 0.14, 0.16]
    
    status_choices = [
        "Open",
        "In Progress",
        "Escalated",
        "Resolved",
        "Closed",
    ]
    status_weights = [0.22, 0.26, 0.16, 0.24, 0.12]
    
    records: List[Dict[str, Any]] = []
    
    for i in range(1, total_records + 1):
        record_id = f"OLA-TCK-{1000 + i}"
        category = rng.choices(category_choices, weights=category_weights, k=1)[0]
        status = rng.choices(status_choices, weights=status_weights, k=1)[0]
        
        days_since_created = rng.randint(0, 30)
        
        if status == "Escalated":
            escalated = True
        else:
            escalation_prob = 0.08 if days_since_created < 15 else 0.18
            escalated = rng.random() < escalation_prob
        
        if category in ["Billing", "General Inquiry"]:
            base_hours = rng.uniform(1.0, 24.0)
        elif category == "Account Access":
            base_hours = rng.uniform(2.0, 48.0)
        else:
            base_hours = rng.uniform(6.0, 96.0)
            
        if escalated:
            base_hours = min(96.0, base_hours * rng.uniform(1.2, 1.8))
        resolution_time_hours = round(max(1.0, min(96.0, base_hours)), 1)
        
        records.append({
            "record_id": record_id,
            "category": category,
            "status": status,
            "resolution_time_hours": resolution_time_hours,
            "days_since_created": days_since_created,
            "escalated": escalated,
        })
        
    return records


# Generate default dataset
SUPPORT_TICKETS: List[Dict[str, Any]] = generate_support_tickets(seed=RANDOM_SEED, total_records=50)


def validate_and_report_dataset(tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates the dataset against all Part 1 Task 1 requirements and prints stats.
    """
    total = len(tickets)
    category_counts: Dict[str, int] = {cat: 0 for cat in CATEGORIES}
    status_counts: Dict[str, int] = {st: 0 for st in STATUSES}
    escalated_count = 0
    
    for t in tickets:
        cat = t["category"]
        st = t["status"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        status_counts[st] = status_counts.get(st, 0) + 1
        if t["escalated"]:
            escalated_count += 1
            
    escalated_pct = (escalated_count / total) * 100.0
    
    print("=" * 60)
    print("OLA SUPPORT TICKET DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Total Records Generated: {total} (Requirement: >= 40)")
    print(f"Seed Used: {RANDOM_SEED}")
    print("\nCounts per Category (Requirement: every category >= 3):")
    for cat, count in sorted(category_counts.items()):
        status_ok = "[OK]" if count >= 3 else "[FAIL]"
        print(f"  - {cat:20s}: {count:2d} records {status_ok}")
        
    print("\nCounts per Status (Requirement: every status >= 1):")
    for st, count in sorted(status_counts.items()):
        status_ok = "[OK]" if count >= 1 else "[FAIL]"
        print(f"  - {st:20s}: {count:2d} records {status_ok}")
        
    status_pct_ok = "[OK]" if 10.0 <= escalated_pct <= 30.0 else "[FAIL]"
    print(f"\nEscalated Records: {escalated_count}/{total} ({escalated_pct:.1f}%)")
    print(f"Escalated Percentage Requirement (10.0% to 30.0%): {status_pct_ok}")
    print("=" * 60)
    
    assert total >= 40, f"Total records {total} < 40"
    for cat in CATEGORIES:
        assert category_counts[cat] >= 3, f"Category {cat} count {category_counts[cat]} < 3"
    for st in STATUSES:
        assert status_counts[st] >= 1, f"Status {st} count {status_counts[st]} < 1"
    assert 10.0 <= escalated_pct <= 30.0, f"Escalated pct {escalated_pct:.1f}% not in [10%, 30%]"
    
    return {
        "total_records": total,
        "category_counts": category_counts,
        "status_counts": status_counts,
        "escalated_count": escalated_count,
        "escalated_percentage": escalated_pct,
    }


if __name__ == "__main__":
    validate_and_report_dataset(SUPPORT_TICKETS)
