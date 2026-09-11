"""
mock_llm.py - Deterministic Mock LLM for CrewAI and Autogen
Track: Business Operations / Customer Support (Ola)

Key Capabilities:
1. Extends crewai.llms.base_llm.BaseLLM for zero-network-access deterministic execution.
2. Avoids Pitfall 1: Ignores the system-prompt template (which contains 'Observation: the result of the action')
   and only parses actual non-system messages for real tool observations.
3. Avoids Pitfall 2: Dispatches tool arguments by inspecting the declared argument schema
   (e.g. 'record_id' vs 'query') rather than substring-matching tool names.
4. Generates valid ReAct steps: Thought -> Action -> Action Input, followed by Thought -> Final Answer.
5. Supports structured Pydantic response formatting.
"""

import sys
import re
import json
import logging
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool


def extract_record_id(text: str) -> Optional[str]:
    """Extracts Ola ticket record ID like OLA-TCK-1001 or OLA-1001."""
    m = re.search(r'\b(OLA(?:-TCK)?-\d{4})\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def extract_observation(messages: Union[str, List[Any]]) -> Optional[str]:
    """
    Safely extracts tool Observation from the conversation.
    CRITICAL (Pitfall 1): Never searches the system prompt template for 'Observation:'.
    Only searches messages originating from the environment/user/assistant after system prompt.
    """
    if isinstance(messages, str):
        # If it's a raw string, find occurrences after the ReAct template preamble
        # The template ends before 'Current Task:' or 'Begin!'
        begin_idx = messages.rfind("Current Task:")
        if begin_idx == -1:
            begin_idx = messages.rfind("Begin!")
        search_region = messages[begin_idx:] if begin_idx != -1 else messages
        
        # Look for 'Observation:' strictly in the execution region
        m = re.search(r'\nObservation:\s*(.*?)(?=\nThought:|\Z)', search_region, re.DOTALL)
        if m:
            return m.group(1).strip()
        return None
        
    elif isinstance(messages, list):
        # Inspect list of message objects or dicts, skipping role == 'system'
        for msg in reversed(messages):
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
            if role == "system":
                continue  # Skip system prompt template entirely
                
            content = getattr(msg, "content", "") or (msg.get("content") if isinstance(msg, dict) else "")
            if isinstance(content, str) and "Observation:" in content:
                m = re.search(r'Observation:\s*(.*?)(?=\nThought:|\Z)', content, re.DOTALL)
                if m:
                    return m.group(1).strip()
                return content.replace("Observation:", "").strip()
                
            # If role is 'tool', content is the direct observation
            if role in ["tool", "function"]:
                return str(content).strip()
                
            # If task context from a prior agent is passed (e.g. to Response Composer)
            if isinstance(content, str) and "This is the context you're working with:" in content:
                ctx_part = content.split("This is the context you're working with:")[1].strip()
                # Check for JSON object from prior task
                try:
                    jm = re.search(r'\{.*\}', ctx_part, re.DOTALL)
                    if jm:
                        cdata = json.loads(jm.group(0))
                        ans = cdata.get("answer") or cdata.get("summary", "")
                        srcs = cdata.get("sources", [])
                        if ans:
                            ans_str = str(ans).strip()
                            if srcs and not any(s in ans_str for s in srcs):
                                return f"{ans_str} (Sources: {', '.join(srcs)})"
                            return ans_str
                except Exception:
                    pass
                clean_ctx = ctx_part.split("Provide your complete response:")[0].strip()
                if clean_ctx:
                    return clean_ctx
                
    return None


def extract_query_text(messages: Union[str, List[Any]]) -> str:
    """Extracts the operational query from the conversation text."""
    full_text = ""
    if isinstance(messages, str):
        full_text = messages
    elif isinstance(messages, list):
        for msg in messages:
            content = getattr(msg, "content", "") or (msg.get("content") if isinstance(msg, dict) else "")
            full_text += " " + str(content)
            
    # Try to extract from 'Current Task:' or 'task:'
    m = re.search(r'(?:Current Task|task):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)', full_text, re.IGNORECASE | re.DOTALL)
    if m:
        task_text = m.group(1).strip()
        # If task text wraps 'for query: <actual_query>', extract the user's actual question
        qm = re.search(r'for query:\s*(.*)', task_text, re.IGNORECASE)
        if qm:
            return qm.group(1).strip()
        return task_text
        
    # Or first user message
    if isinstance(messages, list):
        for msg in messages:
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
            if role == "user":
                c = getattr(msg, "content", "") or (msg.get("content") if isinstance(msg, dict) else "")
                return str(c).strip()
                
    return full_text.strip()[:200]


class OlaCrewBaseLLM(BaseLLM):
    """
    CrewAI BaseLLM implementation for zero-network-access deterministic execution.
    """
    model: str = "mock-ola-support-llm"
    llm_type: str = "custom"

    def __init__(self, **kwargs):
        super().__init__(model=kwargs.get("model", "mock-ola-support-llm"), **kwargs)

    def call(
        self,
        messages: Union[str, List[Any]],
        tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        response_model: Optional[type[BaseModel]] = None,
    ) -> Union[str, Any]:
        """
        Deterministic ReAct execution loop handling tool calls and final answers.
        """
        obs = extract_observation(messages)
        query = extract_query_text(messages)
        ticket_id = extract_record_id(query)
        
        agent_role = getattr(from_agent, "role", "") if from_agent else ""
        agent_tools = getattr(from_agent, "tools", []) if from_agent else (tools or [])
        
        # 1. If an observation exists in the conversation, return the Final Answer
        if obs:
            return self._compose_final_answer(query, obs, response_model, agent_role)
            
        # 2. If the agent has tools, determine which tool to call based on ARGUMENT SCHEMA (Pitfall 2)
        if agent_tools:
            tool_to_call = agent_tools[0]
            
            # Inspect argument schema of the tool
            args_schema = getattr(tool_to_call, "args_schema", None)
            field_names = list(args_schema.model_fields.keys()) if args_schema else []
            
            tool_name = getattr(tool_to_call, "name", "tool")
            
            if "record_id" in field_names:
                # Dispatched because schema declares 'record_id'
                rec_id = ticket_id or "OLA-TCK-1001"
                action_input = json.dumps({"record_id": rec_id})
                thought = f"I need to check the status of ticket {rec_id} using the ticket status tool."
            elif "query" in field_names:
                # Dispatched because schema declares 'query'
                action_input = json.dumps({"query": query})
                thought = f"I need to retrieve knowledge base policy context for query: {query[:60]}."
            else:
                action_input = json.dumps({"query": query})
                thought = "I need to invoke the tool with the provided input."
                
            return (
                f"Thought: {thought}\n"
                f"Action: {tool_name}\n"
                f"Action Input: {action_input}"
            )
            
        # 3. If agent has no tools (e.g. Response Composer), synthesize directly
        return self._compose_final_answer(query, None, response_model, agent_role)

    def _compose_final_answer(
        self,
        query: str,
        obs: Optional[str],
        response_model: Optional[type[BaseModel]],
        agent_role: str = ""
    ) -> str:
        ticket_id = extract_record_id(query)
        
        # If response_model is provided (or if requested as structured response)
        is_ticket_query = bool(ticket_id or (obs and "OLA-TCK-" in obs))
        
        if is_ticket_query:
            rec_id = ticket_id or "OLA-TCK-1001"
            clean_obs = obs.strip() if obs else ""
            clean_obs = re.sub(r'^(?:Support Ticket Status for [^:]+:\s*The ticket has been looked up in the Ola support database\.\s*Details:\s*)+', '', clean_obs).strip()
            clean_obs = re.sub(r'\s*\(Sources?:.*?\)', '', clean_obs).strip()
            
            answer_text = (
                f"Hello! Here is the live status report for support ticket {rec_id} from the Ola support database:\n\n"
                f"• Details: {clean_obs if clean_obs else 'Status retrieved and validated against active resolution metrics.'}\n\n"
                f"Our operations team monitors ticket lifecycles 24/7. Please let us know if you require further assistance with this ticket."
            )
            response_type = "ticket_status"
            sources = ["SUPPORT_TICKETS_DB"]
            escalation_recommended = bool(obs and "RECOMMENDED" in obs)
            ticket_details = {"record_id": rec_id, "summary": clean_obs or obs}
        else:
            if obs:
                clean_obs = obs.strip()
                while clean_obs.startswith("According to Ola Support Policy:"):
                    clean_obs = clean_obs[len("According to Ola Support Policy:"):].strip()
                kb_matches = re.findall(r'OLA-KB-\d{3}', obs)
                clean_obs = re.sub(r'\s*\(Sources?:.*?\)', '', clean_obs).strip()
                
                # Contextual extraction from query / uploaded receipt
                booking_m = re.search(r'(?:Booking\s*(?:ID)?|CRN)[\s:#]+([#A-Za-z0-9-_]+)', query, re.IGNORECASE)
                booking_id = booking_m.group(1).strip() if booking_m else None
                driver_m = re.search(r'Driver:\s*([A-Za-z\s\.]+?)(?:\s*\(|\n|,|$)', query, re.IGNORECASE)
                driver_name = driver_m.group(1).strip() if driver_m else None
                dest_m = re.search(r'Destination[\s:]+([A-Za-z0-9\s]+?)(?:\s*Date|\n|$)', query, re.IGNORECASE)
                destination = dest_m.group(1).strip() if dest_m else None

                lower_q = query.lower()
                lines = []
                
                # Conversational Greeting
                if booking_id:
                    ride_ctx = f"Booking {booking_id}"
                    if driver_name:
                        ride_ctx += f" with driver {driver_name}"
                    if destination:
                        ride_ctx += f" to {destination}"
                    lines.append(f"Hello! Thank you for reaching out to Ola Customer Support regarding your trip ({ride_ctx}).")
                else:
                    lines.append("Hello! Thank you for contacting Ola Customer Support.")
                    
                lines.append("")
                
                # Contextual Policy Guidance
                if any(k in lower_q for k in ["refund", "eligible", "money back", "fare", "overcharge", "cancellation", "cancel"]):
                    lines.append("Here is the eligibility evaluation for your request under official Ola support policy:")
                    lines.append("")
                    if "72 hours" in clean_obs or "telemetry" in clean_obs.lower():
                        lines.append("• Eligibility & Telemetry Verification: Full or partial fare refunds are approved for verified service failures, including driver cancellations after arriving late, incorrect toll fee additions, or severe route deviations. Refund requests submitted within 72 hours of trip completion are evaluated automatically against vehicle GPS trip telemetry and driver dispatch records.")
                    elif clean_obs:
                        lines.append(f"• Eligibility Policy: {clean_obs}")
                        
                    if "3 to 5 banking days" in clean_obs or "ola money" in clean_obs.lower():
                        lines.append("• Disbursement Timeline: Approved refunds are credited to your original payment source within 3 to 5 banking days, or deposited instantly to your Ola Money wallet balance at your discretion.")
                        
                    if "250" in clean_obs or "inconvenience" in clean_obs.lower():
                        lines.append("• Inconvenience Compensation: In severe disruption cases where a rider is stranded due to verified driver refusal, an inconvenience compensation credit of up to ₹250 may be authorized by an L2 specialist.")
                        
                    lines.append("")
                    claim_ref = f"Booking {booking_id}" if booking_id else "this ride"
                    lines.append(f"Next Steps: If you encountered a service issue on {claim_ref}, you can submit your dispute directly through the Ola app under 'Ride History' > select this trip > 'Report an Issue with Fare/Driver'. Our support team operates 24/7 to assist you.")

                elif any(k in lower_q for k in ["sla", "response time", "severity", "p1", "p2"]):
                    lines.append("Here is the official Service Level Agreement (SLA) turnaround policy by severity tier:")
                    lines.append(f"\n{clean_obs}\n")
                    lines.append("Our operations dispatch team monitors active incidents 24/7 to ensure timely resolution.")

                elif any(k in lower_q for k in ["repeat", "second time", "same issue"]):
                    lines.append("Regarding repeat complaint handling:")
                    lines.append(f"\n{clean_obs}\n")
                    lines.append("Repeat complaints are automatically routed for priority review by Senior Operations Specialists.")

                else:
                    lines.append(f"According to Ola Support Policy: {clean_obs}")
                    lines.append("\nPlease let us know if you need any additional assistance with your ride or account.")

                answer_text = "\n".join(lines)
                sources = list(dict.fromkeys(kb_matches)) if kb_matches else ["OLA-KB-CORE"]
            else:
                qm = re.search(r'for query:\s*(.*)', query, re.IGNORECASE)
                clean_q = qm.group(1).strip() if qm else query.strip()
                answer_text = (
                    f"Hello! Thank you for reaching out to Ola Customer Support. Regarding your inquiry '{clean_q[:80]}':\n\n"
                    f"I do not have sufficient information in the official Ola knowledge base to assist with this specific topic. "
                    f"Please contact our frontline support desk for specialized assistance."
                )
                sources = ["OLA-KB-CORE"]
            response_type = "policy_inquiry"
            escalation_recommended = False
            ticket_details = None
            
        structured_data = {
            "query": query,
            "response_type": response_type,
            "answer": answer_text,
            "sources": sources,
            "confidence": 0.95,
            "escalation_recommended": escalation_recommended,
            "ticket_details": ticket_details,
        }
        
        if response_model is not None:
            return json.dumps(structured_data)
            
        # For standard ReAct agent return
        return (
            f"Thought: I have all required information to synthesize the final verified response.\n"
            f"Final Answer: {json.dumps(structured_data)}"
        )
