"""
interactive_chat.py - Plain Text Interactive Chat for Ola Domain Support Agent
Run this script to chat directly with the support agent in your terminal without JSON or Swagger UI!
"""

import sys
import uuid
from app import ask_endpoint, AskRequest

def main():
    print("=" * 70)
    print("   OLA DOMAIN SUPPORT AGENT - INTERACTIVE TERMINAL CHAT")
    print("=" * 70)
    print("Ask any policy, SLA, refund, or ticket status question in plain English.")
    print("Commands:")
    print("  'new'   - Start a fresh conversation session")
    print("  'exit'  - Quit the chat")
    print("=" * 70)
    
    session_id = f"user-session-{str(uuid.uuid4())[:8]}"
    print(f"Current Session: {session_id}\n")
    
    while True:
        try:
            query = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break
            
        if not query:
            continue
            
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting chat. Goodbye!")
            break
            
        if query.lower() == "new":
            session_id = f"user-session-{str(uuid.uuid4())[:8]}"
            print(f"\n[Started new fresh session: {session_id}]")
            continue
            
        print("\nAgent is thinking...")
        req = AskRequest(query=query, session_id=session_id)
        resp = ask_endpoint(req)
        
        print("\n" + "-" * 70)
        print(f"Response Type: {resp.response_type.upper()}")
        print(f"Sources:       {', '.join(resp.sources) if resp.sources else 'None'}")
        print(f"Status:        {'APPROVED' if resp.autogen_approved else 'FLAGGED'}")
        print("-" * 70)
        print(f"\nAgent > {resp.answer}\n")
        print("-" * 70)

if __name__ == "__main__":
    main()
