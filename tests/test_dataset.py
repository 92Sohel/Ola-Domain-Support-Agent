"""
tests/test_dataset.py - Validation Tests for Ola Support Ticket Dataset Generator
Track: Business Operations / Customer Support (Ola)

Verifies Part 1 Task 1:
- >=40 records generated
- Every category in vocabulary has >=3 records
- Every status in vocabulary has >=1 record
- Escalated percentage lands strictly within [10.0%, 30.0%]
- Determinism with seed=42
- Realistic resolution time range
"""

import pytest
from dataset import (
    generate_support_tickets,
    validate_and_report_dataset,
    CATEGORIES,
    STATUSES,
    RANDOM_SEED
)


def test_dataset_generation_constraints():
    tickets = generate_support_tickets(seed=RANDOM_SEED, total_records=50)
    report = validate_and_report_dataset(tickets)
    
    assert report["total_records"] == 50
    assert report["total_records"] >= 40
    
    # Check category counts
    for cat in CATEGORIES:
        assert report["category_counts"][cat] >= 3, f"Category {cat} has less than 3 records"
        
    # Check status counts
    for st in STATUSES:
        assert report["status_counts"][st] >= 1, f"Status {st} has less than 1 record"
        
    # Check escalated percentage
    pct = report["escalated_percentage"]
    assert 10.0 <= pct <= 30.0, f"Escalated percentage {pct}% not in [10%, 30%]"
    
    # Check resolution times and days
    for t in tickets:
        assert 1.0 <= t["resolution_time_hours"] <= 96.0
        assert 0 <= t["days_since_created"] <= 30
        assert isinstance(t["escalated"], bool)


def test_dataset_reproducibility():
    draw1 = generate_support_tickets(seed=RANDOM_SEED, total_records=50)
    draw2 = generate_support_tickets(seed=RANDOM_SEED, total_records=50)
    assert draw1 == draw2, "Dataset generator must be deterministic with identical seed"
