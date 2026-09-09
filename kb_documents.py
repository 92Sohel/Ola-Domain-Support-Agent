"""
kb_documents.py - Knowledge Base for Ola Domain Support Operations
Track: Business Operations / Customer Support (Ola)

Covers all 12 required policy topics:
1. ticket-priority classification rules
2. SLA-by-severity policy
3. escalation matrix
4. refund/compensation policy
5. customer-communication-channel policy
6. business-hours/holiday-support policy
7. repeat-complaint-handling policy
8. service-credit policy
9. feedback-collection process
10. VIP-customer handling policy
11. outage-communication protocol
12. data-retention policy for tickets
"""

from typing import List, Dict

KB_DOCUMENTS: List[Dict[str, str]] = [
    {
        "doc_id": "OLA-KB-001",
        "topic": "ticket-priority classification rules",
        "title": "Ola Support Ticket Priority Classification Rules",
        "content": (
            "Ola support tickets are systematically classified into four operational priority tiers: P1-Critical, P2-High, P3-Medium, and P4-Low. "
            "P1-Critical priority is reserved strictly for passenger safety hazards, in-trip physical emergencies, SOS alert activations, and vehicle accident incidents. "
            "P2-High encompasses active trip stoppages, driver no-shows with stranded passengers, fare calculation overcharges exceeding ₹500, and critical app crash loops during ride booking. "
            "P3-Medium covers post-ride fare disputes below ₹500, driver behavior feedback, route deviation disputes, and standard account verification delays. "
            "P4-Low addresses general informational queries, promotional code inquiries, lost-and-found status follow-ups, and profile information update requests."
        )
    },
    {
        "doc_id": "OLA-KB-002",
        "topic": "SLA-by-severity policy",
        "title": "Ola Support Service Level Agreement (SLA) by Severity",
        "content": (
            "Ola enforces strict Service Level Agreements across all support channels to guarantee timely resolution based on ticket severity. "
            "P1-Critical incidents require a frontline response within 15 minutes and a targeted resolution or safety containment within 2 hours. "
            "P2-High severity tickets mandate a human or senior agent acknowledgment within 1 hour and full operational resolution within 8 hours. "
            "P3-Medium tickets carry a 4-hour initial response window with complete dispute investigation and closure within 24 hours. "
            "P4-Low inquiries must receive an acknowledgment within 12 hours and final informational resolution within 48 business hours."
        )
    },
    {
        "doc_id": "OLA-KB-003",
        "topic": "escalation matrix",
        "title": "Ola Customer Support Multi-Tier Escalation Matrix",
        "content": (
            "The Ola customer support escalation matrix structures ticket routing across Level 1 (L1) Frontline Specialists, Level 2 (L2) Senior Operations Specialists, and Level 3 (L3) Engineering and Executive Grievance Leads. "
            "Tickets remain with L1 frontline support if resolved within the designated SLA window using standard operational playbooks. "
            "A ticket automatically escalates to L2 whenever an SLA breach occurs, a customer challenges a fare dispute twice, or a complex technical malfunction is flagged. "
            "L3 escalation is immediately triggered for unresolved safety investigations, payment gateway batch failures, legal notices, or issues affecting VIP riders."
        )
    },
    {
        "doc_id": "OLA-KB-004",
        "topic": "refund/compensation policy",
        "title": "Ola Rider Refund and Financial Compensation Policy",
        "content": (
            "Riders are entitled to full or partial fare refunds when service failures occur, including driver cancellations after arriving late, incorrect toll fee additions, or severe route deviations. "
            "Refund requests submitted within 72 hours of trip completion are evaluated automatically against GPS trip telemetry and driver dispatch records. "
            "Approved refunds are credited to the original payment source within 3 to 5 banking days, or deposited instantly to the user's Ola Money wallet balance at the rider's discretion. "
            "In severe disruption cases where a rider is stranded due to verified driver refusal, a inconvenience compensation credit of up to ₹250 may be authorized by an L2 specialist."
        )
    },
    {
        "doc_id": "OLA-KB-005",
        "topic": "customer-communication-channel policy",
        "title": "Ola Customer Communication Channel and Notification Protocol",
        "content": (
            "Ola provides customer assistance exclusively through verified official channels: the in-app 24/7 Support Chat, the dedicated Emergency Safety Helpline, registered email support, and automated SMS alerts. "
            "Support representatives will never contact riders or drivers via personal WhatsApp numbers or request debit card PINs, CVV codes, or one-time passwords (OTPs). "
            "All ticket progress updates and resolution summaries are delivered synchronously via in-app message notifications and backed up by email transcripts. "
            "For active safety alerts triggered via the in-app SOS button, outbound phone calls from the Ola Safety Response Team take immediate precedence over digital text communications."
        )
    },
    {
        "doc_id": "OLA-KB-006",
        "topic": "business-hours/holiday-support policy",
        "title": "Ola Business Hours, Shift Coverage, and Holiday Support Policy",
        "content": (
            "Ola operates a continuous 24/7/365 support infrastructure for all active rides, driver partner road emergencies, and passenger safety incidents across all operating cities. "
            "General billing inquiries, post-trip audit reviews, driver document onboarding, and non-urgent account updates are processed during standard business hours from 08:00 to 22:00 IST Monday through Sunday. "
            "During national holidays and regional festival periods, core technical and safety teams maintain full 100% operational staffing while administrative back-office resolution times may experience an extended 12-hour grace period. "
            "Automated self-serve resolution workflows within the Ola app remain fully accessible at all times regardless of holidays."
        )
    },
    {
        "doc_id": "OLA-KB-007",
        "topic": "repeat-complaint-handling policy",
        "title": "Ola Repeat Complaint and Recurrent Issue Handling Policy",
        "content": (
            "A support ticket is tagged as a 'Repeat Complaint' if a rider or driver partner logs two or more inquiries regarding the same ride ID or root problem within a 7-day rolling window. "
            "Repeat complaints bypass Level 1 frontline scripts immediately and are reassigned directly to a designated Level 2 Senior Resolution Specialist. "
            "The assigned L2 specialist must review the complete prior conversation history and telemetry logs before contacting the user with a comprehensive resolution within 4 hours. "
            "If a customer files three consecutive complaints regarding driver behavior or app payment errors, an internal incident review is automatically dispatched to the City Fleet Operations team."
        )
    },
    {
        "doc_id": "OLA-KB-008",
        "topic": "service-credit policy",
        "title": "Ola Promotional Service Credit and Wallet Disbursement Policy",
        "content": (
            "Service credits are issued to customer accounts as goodwill gestures or operational remediation for minor ride delays, cleaniness complaints, or vehicle AC malfunctions. "
            "Credits are credited directly to the customer's Ola Money balance in predetermined denominations of ₹50, ₹100, or ₹200 based on the severity of the service disruption. "
            "All promotional service credits remain valid for exactly 12 months from the date of issuance and are automatically applied against the fare of subsequent Ola bookings. "
            "Service credits cannot be liquidated, transferred between independent user accounts, or withdrawn to external bank accounts."
        )
    },
    {
        "doc_id": "OLA-KB-009",
        "topic": "feedback-collection process",
        "title": "Ola Rider and Driver Post-Trip Feedback Collection Process",
        "content": (
            "Upon completion of every trip, riders and drivers are prompted to submit a star rating from 1 to 5 stars along with optional categorical feedback tags such as car hygiene, route driving, or polite conduct. "
            "Any trip receiving a 1-star or 2-star rating instantly triggers an automated diagnostic prompt asking the user to specify the root grievance. "
            "Feedback scores below 3 stars generate a proactive low-CSAT review ticket in the customer experience queue, prompting outreach within 24 hours if safety or billing anomalies are noted. "
            "Aggregated weekly driver ratings directly influence driver quality tiers, incentive bonuses, and retraining mandates."
        )
    },
    {
        "doc_id": "OLA-KB-010",
        "topic": "VIP-customer handling policy",
        "title": "Ola Select and Corporate VIP Customer Handling Policy",
        "content": (
            "Ola Select subscribers and registered Corporate Enterprise riders are categorized as VIP accounts and receive prioritized routing across all support queues. "
            "Inbound inquiries from VIP accounts are routed directly to dedicated Senior Relationship Managers with a guaranteed first-response SLA of under 5 minutes. "
            "Disputed charges on VIP accounts under ₹1,000 are eligible for instant provisional credit while the background telematics investigation is conducted. "
            "VIP escalations receive senior management visibility, and any service failure affecting a corporate account triggers a root-cause summary sent to the enterprise account manager."
        )
    },
    {
        "doc_id": "OLA-KB-011",
        "topic": "outage-communication protocol",
        "title": "Ola Platform Technical Outage and Incident Communication Protocol",
        "content": (
            "When widespread technical disruptions affect the Ola booking engine, payment gateways, or driver dispatch systems, the Incident Management Team initiates the Outage Protocol. "
            "A prominent status advisory banner is deployed across the home screen of the rider and driver partner mobile apps within 10 minutes of incident confirmation. "
            "Public communication updates are broadcast across official social channels and SMS every 30 minutes until core service restoration is verified. "
            "During an active platform outage, automated cancellation fees are globally suppressed across all affected zones, and support agents switch to outage containment macros."
        )
    },
    {
        "doc_id": "OLA-KB-012",
        "topic": "data-retention policy for tickets",
        "title": "Ola Support Ticket and Customer Data Retention Policy",
        "content": (
            "Support ticket records, chat transcripts, and recorded telephonic customer interactions are retained in active storage for 3 years from the date of ticket closure for dispute resolution and auditing. "
            "In compliance with the Indian Digital Personal Data Protection (DPDP) Act and internal governance guidelines, customer phone numbers and payment tokens are automatically masked after 180 days of ticket inactivity. "
            "After the 3-year statutory audit period expires, raw customer support logs are permanently purged or irreversibly anonymized for operational machine learning and aggregate analytics. "
            "Customers maintain the right to request a certified copy or erasure of their personal support history, subject to mandatory fraud prevention and ongoing legal holds."
        )
    }
]


def get_all_documents() -> List[Dict[str, str]]:
    """Returns the complete list of 12 Ola Knowledge Base policy documents."""
    return KB_DOCUMENTS


if __name__ == "__main__":
    print(f"Loaded {len(KB_DOCUMENTS)} Ola Knowledge Base documents.")
    for doc in KB_DOCUMENTS:
        sentences = [s for s in doc["content"].split(". ") if s.strip()]
        print(f"- [{doc['doc_id']}] {doc['topic']}: {len(sentences)} sentences")
