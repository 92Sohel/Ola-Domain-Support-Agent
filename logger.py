"""
logger.py - ELK-Style Structured JSON-Lines Logger with PII Sanitization
Track: Business Operations / Customer Support (Ola)

Requirements:
- Logs every request as one JSON-Lines entry
- Includes trace_id, session_id, timestamp, duration_ms, status
- Zero Disk PII Guarantee: Raw unmasked phone numbers are stripped/masked before writing to disk
"""

import os
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from guardrails import mask_pii

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
AUDIT_LOG_FILE = os.path.join(LOGS_DIR, "audit.jsonl")

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)


class StructuredLogger:
    def __init__(self, log_filepath: str = AUDIT_LOG_FILE):
        self.log_filepath = log_filepath

    def log_request(
        self,
        endpoint: str,
        raw_query: str,
        raw_response: Any,
        duration_ms: float,
        status: str = "SUCCESS",
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Logs a structured request event.
        Guarantees that raw_query and raw_response are sanitized of fixed-format PII before writing to disk.
        """
        if not trace_id:
            trace_id = str(uuid.uuid4())
            
        if not session_id:
            session_id = "default-session"
            
        # PII Sanitization: NEVER write raw unmasked PII to disk
        sanitized_query, _ = mask_pii(str(raw_query))
        
        if isinstance(raw_response, dict):
            # Recursively or string-based mask
            resp_str = json.dumps(raw_response)
            sanitized_resp_str, _ = mask_pii(resp_str)
            sanitized_response = json.loads(sanitized_resp_str)
        else:
            sanitized_response, _ = mask_pii(str(raw_response))
            
        log_entry = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "endpoint": endpoint,
            "duration_ms": round(duration_ms, 2),
            "status": status,
            "sanitized_query": sanitized_query,
            "sanitized_response": sanitized_response,
            "metadata": extra_metadata or {},
        }
        
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        return log_entry


# Singleton logger instance
AUDIT_LOGGER = StructuredLogger()


if __name__ == "__main__":
    print("Testing StructuredLogger:")
    test_query = "Please update ticket for phone +91 9123456789."
    test_resp = {"message": "Confirmed update for contact +91 9123456789"}
    
    entry = AUDIT_LOGGER.log_request(
        endpoint="/ask",
        raw_query=test_query,
        raw_response=test_resp,
        duration_ms=45.2,
        status="SUCCESS",
        session_id="session-test-1",
    )
    
    print("Logged Entry:")
    print(json.dumps(entry, indent=2))
    
    # Verify disk log does NOT contain the raw phone number
    with open(AUDIT_LOGGER.log_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        last_line = lines[-1]
        assert "9123456789" not in last_line, "Security Violation: Raw phone number found in disk log!"
        assert "[REDACTED_PHONE]" in last_line, "Masked token missing from disk log!"
        
    print("\nStructured logging test passed: Zero disk PII confirmed!")
