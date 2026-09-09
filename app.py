"""
app.py - Production FastAPI Deployment for Ola Domain Support Agent
Track: Business Operations / Customer Support (Ola)

Requirements (Part 3 Tasks 11 & 12):
- >=2 HTTP endpoints: POST /ask, POST /add-document
- WebSocket endpoint: @app.websocket("/ws/chat/{session_id}") for real-time multi-turn chat
- Gracefully handles WebSocketDisconnect (keeps server running for other clients)
- Structured ELK JSON-Lines logging with Trace ID & timing info
- Zero Disk PII: raw phone numbers are masked before any log entry reaches disk
- Full pipeline: Guardrails -> Cost Budget -> Cache -> CrewAI -> Autogen Review -> Structured Output
"""

import sys
import time
import uuid
import json
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from guardrails import run_input_guardrail
from governance import enforce_runtime_budget, RuntimeBudgetExceededError
from cache import GLOBAL_CACHE
from crew_agent import process_chat_turn, CrewResponse, get_rag_core
from review_stage import review_draft_sync
from logger import AUDIT_LOGGER
from rag_core import chunk_fixed_size_overlap, chunk_sentence_based


# ==========================================
# Pydantic Request / Response Models
# ==========================================

class AskRequest(BaseModel):
    query: str = Field(..., description="Customer or operations support query")
    session_id: Optional[str] = Field("default-session", description="Unique conversation session ID")
    trace_id: Optional[str] = Field(None, description="Optional distributed trace ID")


class AskResponse(BaseModel):
    query: str
    response_type: str
    answer: str
    sources: List[str]
    confidence: float
    escalation_recommended: bool
    ticket_details: Optional[Dict[str, Any]] = None
    reviewed_by_autogen: bool = True
    autogen_approved: bool = True
    autogen_review_reason: str = "Verified compliant."
    cached: bool = False
    trace_id: str
    duration_ms: float


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., description="Unique document ID, e.g. OLA-KB-013")
    title: str = Field(..., description="Document title")
    topic: str = Field(..., description="Knowledge domain topic")
    content: str = Field(..., description="2-5 sentences of policy text")


class AddDocumentResponse(BaseModel):
    success: bool
    doc_id: str
    fixed_chunks_added: int
    sentence_chunks_added: int
    message: str
    trace_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up RAG core
    get_rag_core()
    yield


app = FastAPI(
    title="Ola Domain Support Agent API",
    description="Enterprise Multi-Agent Customer & Operations Support for Ola",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# HTTP Endpoints
# ==========================================

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "HEALTHY", "service": "Ola Domain Support Agent", "mock_mode": True}


@app.post("/ask", response_model=AskResponse, tags=["Support"])
def ask_endpoint(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    trace_id = req.trace_id or str(uuid.uuid4())
    session_id = req.session_id or "default-session"
    
    # 1. Input Guardrails: PII Masking and Prompt Injection Detection
    guard_res = run_input_guardrail(req.query)
    sanitized_query = guard_res["sanitized_text"]
    
    if guard_res["blocked"]:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        resp = AskResponse(
            query=sanitized_query,
            response_type="refusal",
            answer=f"Request blocked by security guardrails: {guard_res['rejection_reason']}",
            sources=[],
            confidence=1.0,
            escalation_recommended=False,
            ticket_details=None,
            reviewed_by_autogen=False,
            autogen_approved=False,
            autogen_review_reason="Blocked at perimeter by input guardrail.",
            cached=False,
            trace_id=trace_id,
            duration_ms=round(duration_ms, 2)
        )
        AUDIT_LOGGER.log_request(
            endpoint="/ask",
            raw_query=req.query,
            raw_response=resp.model_dump(),
            duration_ms=duration_ms,
            status="BLOCKED_GUARDRAIL",
            session_id=session_id,
            trace_id=trace_id
        )
        return resp
        
    # 2. Runtime Budget Enforcement
    try:
        enforce_runtime_budget(sanitized_query)
    except RuntimeBudgetExceededError as e:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        AUDIT_LOGGER.log_request(
            endpoint="/ask",
            raw_query=req.query,
            raw_response={"error": str(e)},
            duration_ms=duration_ms,
            status="REJECTED_BUDGET",
            session_id=session_id,
            trace_id=trace_id
        )
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))
        
    # 3. In-Memory Response Caching Check (Session-aware to prevent cross-session memory pollution)
    cache_key = f"{session_id}:{sanitized_query}"
    cached_entry = GLOBAL_CACHE.get(cache_key)
    if cached_entry:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        cached_resp = AskResponse(
            **cached_entry,
            cached=True,
            trace_id=trace_id,
            duration_ms=round(duration_ms, 2)
        )
        AUDIT_LOGGER.log_request(
            endpoint="/ask",
            raw_query=req.query,
            raw_response=cached_resp.model_dump(),
            duration_ms=duration_ms,
            status="SUCCESS_CACHE_HIT",
            session_id=session_id,
            trace_id=trace_id
        )
        return cached_resp
        
    # 4. CrewAI Crew Multi-Agent Execution with Session Memory
    crew_output: CrewResponse = process_chat_turn(session_id, sanitized_query)
    
    # 5. Autogen Review Stage (Policy Reviewer + Final Editor)
    verdict = review_draft_sync(
        draft_answer=crew_output.answer,
        retrieved_context=f"Sources: {crew_output.sources}"
    )
    
    final_answer = verdict.final_answer
    
    duration_ms = (time.perf_counter() - t0) * 1000.0
    
    resp_data = {
        "query": sanitized_query,
        "response_type": crew_output.response_type,
        "answer": final_answer,
        "sources": crew_output.sources,
        "confidence": crew_output.confidence,
        "escalation_recommended": crew_output.escalation_recommended,
        "ticket_details": crew_output.ticket_details,
        "reviewed_by_autogen": True,
        "autogen_approved": verdict.approved,
        "autogen_review_reason": verdict.reason,
    }
    
    # Save to response cache (keyed by session and query)
    GLOBAL_CACHE.set(cache_key, resp_data)
    
    full_resp = AskResponse(
        **resp_data,
        cached=False,
        trace_id=trace_id,
        duration_ms=round(duration_ms, 2)
    )
    
    # 6. Structured ELK Logging with Zero Disk PII Guarantee
    AUDIT_LOGGER.log_request(
        endpoint="/ask",
        raw_query=req.query,
        raw_response=full_resp.model_dump(),
        duration_ms=duration_ms,
        status="SUCCESS",
        session_id=session_id,
        trace_id=trace_id
    )
    
    return full_resp


@app.post("/add-document", response_model=AddDocumentResponse, tags=["Knowledge Base"])
def add_document_endpoint(doc: AddDocumentRequest) -> AddDocumentResponse:
    t0 = time.perf_counter()
    trace_id = str(uuid.uuid4())
    
    rag = get_rag_core()
    doc_dict = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "topic": doc.topic,
        "content": doc.content,
    }
    
    fixed_chunks = chunk_fixed_size_overlap(doc_dict)
    sent_chunks = chunk_sentence_based(doc_dict)
    
    # Index into fixed collection
    f_texts = [c["text"] for c in fixed_chunks]
    f_embs = rag.embed_model.encode(f_texts, convert_to_numpy=True).tolist()
    rag.fixed_collection.upsert(
        ids=[c["chunk_id"] for c in fixed_chunks],
        documents=f_texts,
        embeddings=f_embs,
        metadatas=[{
            "doc_id": c["doc_id"],
            "topic": c["topic"],
            "title": c["title"],
            "strategy": c["strategy"],
            "chunk_index": c["chunk_index"],
        } for c in fixed_chunks]
    )
    
    # Index into sentence collection
    s_texts = [c["text"] for c in sent_chunks]
    s_embs = rag.embed_model.encode(s_texts, convert_to_numpy=True).tolist()
    rag.sentence_collection.upsert(
        ids=[c["chunk_id"] for c in sent_chunks],
        documents=s_texts,
        embeddings=s_embs,
        metadatas=[{
            "doc_id": c["doc_id"],
            "topic": c["topic"],
            "title": c["title"],
            "strategy": c["strategy"],
            "chunk_index": c["chunk_index"],
        } for c in sent_chunks]
    )
    
    duration_ms = (time.perf_counter() - t0) * 1000.0
    
    resp = AddDocumentResponse(
        success=True,
        doc_id=doc.doc_id,
        fixed_chunks_added=len(fixed_chunks),
        sentence_chunks_added=len(sent_chunks),
        message=f"Document '{doc.doc_id}' successfully indexed into both ChromaDB collections.",
        trace_id=trace_id,
    )
    
    AUDIT_LOGGER.log_request(
        endpoint="/add-document",
        raw_query=doc.doc_id,
        raw_response=resp.model_dump(),
        duration_ms=duration_ms,
        status="SUCCESS",
        trace_id=trace_id
    )
    return resp


# ==========================================
# WebSocket Endpoint: Real-time Multi-Turn Chat
# ==========================================

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat_endpoint(websocket: WebSocket, session_id: str):
    """
    Real-time interactive multi-turn chat over WebSocket.
    Maintains session history and gracefully catches WebSocketDisconnect
    to ensure the server continues running smoothly for other active clients.
    """
    await websocket.accept()
    await websocket.send_json({
        "event": "connected",
        "session_id": session_id,
        "message": f"Connected to Ola Support Agent session '{session_id}'."
    })
    
    try:
        while True:
            # Receive query from client
            raw_text = await websocket.receive_text()
            if not raw_text.strip():
                continue
                
            try:
                # Handle JSON payload or raw text
                msg_data = json.loads(raw_text)
                query = msg_data.get("query", raw_text)
            except Exception:
                query = raw_text
                
            # Process query through pipeline
            guard_res = run_input_guardrail(query)
            if guard_res["blocked"]:
                await websocket.send_json({
                    "event": "message",
                    "session_id": session_id,
                    "answer": f"Request blocked by guardrails: {guard_res['rejection_reason']}",
                    "response_type": "refusal",
                    "escalation_recommended": False
                })
                continue
                
            # CrewAI multi-turn processing via thread pool to avoid blocking async event loop
            import asyncio
            crew_output = await asyncio.to_thread(process_chat_turn, session_id, guard_res["sanitized_text"])
            
            # Send response back to client
            await websocket.send_json({
                "event": "message",
                "session_id": session_id,
                "query": guard_res["sanitized_text"],
                "response_type": crew_output.response_type,
                "answer": crew_output.answer,
                "sources": crew_output.sources,
                "escalation_recommended": crew_output.escalation_recommended,
                "ticket_details": crew_output.ticket_details,
            })
            
    except WebSocketDisconnect:
        # Graceful disconnect handling: logs and exits coroutine without crashing the server
        print(f"[WebSocket] Client disconnected cleanly for session: '{session_id}'. Server running.")
    except Exception as e:
        print(f"[WebSocket] Error during session '{session_id}': {e}")
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
