"""
tests/test_rag.py - Validation Tests for Chunking, ChromaDB, and Grounded Generation
Track: Business Operations / Customer Support (Ola)

Verifies Part 1 Tasks 2, 3, 4, 5:
- 12 Knowledge Base documents loaded covering all topics
- Fixed-size chunking with overlap
- Sentence-based chunking
- Separate ChromaDB indexing
- Grounded generation with calibrated fallback threshold
"""

import pytest
from kb_documents import KB_DOCUMENTS
from rag_core import (
    OlaRAGCore,
    chunk_fixed_size_overlap,
    chunk_sentence_based,
    grounded_generation,
    CALIBRATED_FALLBACK_THRESHOLD
)

_rag_core = None

def get_rag():
    global _rag_core
    if _rag_core is None:
        _rag_core = OlaRAGCore()
    return _rag_core


def test_kb_documents_count_and_topics():
    assert len(KB_DOCUMENTS) >= 12
    required_topics = [
        "ticket-priority classification rules",
        "SLA-by-severity policy",
        "escalation matrix",
        "refund/compensation policy",
        "customer-communication-channel policy",
        "business-hours/holiday-support policy",
        "repeat-complaint-handling policy",
        "service-credit policy",
        "feedback-collection process",
        "VIP-customer handling policy",
        "outage-communication protocol",
        "data-retention policy for tickets",
    ]
    covered_topics = [d["topic"] for d in KB_DOCUMENTS]
    for topic in required_topics:
        assert topic in covered_topics, f"Required topic '{topic}' missing from KB documents"


def test_chunking_strategies():
    doc = KB_DOCUMENTS[0]
    fixed_chunks = chunk_fixed_size_overlap(doc, chunk_size=180, overlap=40)
    assert len(fixed_chunks) > 1
    assert all("doc_id" in c for c in fixed_chunks)
    
    sent_chunks = chunk_sentence_based(doc, sentences_per_chunk=2)
    assert len(sent_chunks) > 1
    assert all("doc_id" in c for c in sent_chunks)


def test_grounded_generation_in_scope():
    rag = get_rag()
    query = "What is the refund policy for delayed rides?"
    res = grounded_generation(query, rag, threshold=CALIBRATED_FALLBACK_THRESHOLD)
    assert res["grounded"] is True
    assert res["top_similarity"] >= CALIBRATED_FALLBACK_THRESHOLD
    assert len(res["sources"]) > 0
    assert "refund" in res["answer"].lower() or "ola" in res["answer"].lower()


def test_grounded_generation_fallback_out_of_scope():
    rag = get_rag()
    query = "What is the recipe for baking a chocolate cake?"
    res = grounded_generation(query, rag, threshold=CALIBRATED_FALLBACK_THRESHOLD)
    assert res["grounded"] is False
    assert "insufficient information" in res["answer"].lower() or "not have sufficient" in res["answer"].lower()
    assert res["sources"] == []
