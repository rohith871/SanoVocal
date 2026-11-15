"""
SanoVocal Deterministic Tool Engine
Provides concrete execution implementations for SanoVocal's 7 tools,
grounded in official scheme whitelist and verified knowledge cards.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CARDS_FILE = DATA / "curated" / "verified_knowledge_cards.jsonl"
SCHEMES_FILE = DATA / "selected_schemes.csv"

# In-memory index of verified knowledge cards
_KNOWLEDGE_CACHE: Dict[str, Dict[str, Any]] = {}

def _get_knowledge_cards() -> Dict[str, Dict[str, Any]]:
    global _KNOWLEDGE_CACHE
    if not _KNOWLEDGE_CACHE:
        if CARDS_FILE.exists():
            with open(CARDS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    card = json.loads(line)
                    _KNOWLEDGE_CACHE[card["scheme_id"]] = card
    return _KNOWLEDGE_CACHE

# Acute emergency medical keywords that trigger strict triage override
EMERGENCY_RED_FLAGS = [
    "chest pain", "breathlessness", "difficulty breathing",
    "severe bleeding", "unconscious", "poison", "heart attack",
    "stroke", "paralysis", "suicide", "severe trauma", "road accident",
    "saas lene me dikkat", "chhati me dard", "behosh"
]

def execute_emergency_screen(symptoms: List[str]) -> Dict[str, Any]:
    matched = []
    for s in symptoms:
        s_lower = str(s).lower()
        for kw in EMERGENCY_RED_FLAGS:
            if kw in s_lower:
                matched.append(kw)
    
    if matched:
        return {
            "status": "EMERGENCY",
            "action": "Immediate medical override triggered. Direct user to call 112 / 108 or go to the nearest emergency room immediately. Cease scheme exploration.",
            "emergency_code": "RED_FLAG_TRIAGE"
        }
    return {
        "status": "SAFE",
        "action": "No emergency red flags identified. Safe to proceed with welfare scheme navigation.",
        "emergency_code": None
    }

def execute_match_schemes(age: int, category: str, income_status: str = "eligible") -> Dict[str, Any]:
    cards = _get_knowledge_cards()
    matched = []
    cat_lower = category.lower().strip()

    # Rule: Senior citizen health coverage (70+ priority rule for Ayushman Vay Vandana)
    if age >= 70 and ("health" in cat_lower or "senior" in cat_lower or "insurance" in cat_lower or "all" in cat_lower):
        if "ayushman_vay_vandana" in cards:
            c = cards["ayushman_vay_vandana"]
            matched.append({"scheme_id": c["scheme_id"], "scheme_name": c["scheme_name"], "category": c["category"], "official_url": c["official_url"]})

    # Domain category matching: prioritize exact matches first, then partial matches
    exact_matches = []
    partial_matches = []
    for sid, c in cards.items():
        if sid == "ayushman_vay_vandana" and age >= 70:
            continue
        c_cat = c["category"].lower()
        c_name = c["scheme_name"].lower()
        if cat_lower == c_cat:
            exact_matches.append({"scheme_id": c["scheme_id"], "scheme_name": c["scheme_name"], "category": c["category"], "official_url": c["official_url"]})
        elif cat_lower in c_cat or cat_lower in c_name:
            partial_matches.append({"scheme_id": c["scheme_id"], "scheme_name": c["scheme_name"], "category": c["category"], "official_url": c["official_url"]})

    matched.extend(exact_matches)
    matched.extend(partial_matches)

    # Default fallback to PM-JAY if health related and no specific match
    if not matched and ("health" in cat_lower or "hospital" in cat_lower or "medical" in cat_lower):
        if "ab_pmjay" in cards:
            c = cards["ab_pmjay"]
            matched.append({"scheme_id": c["scheme_id"], "scheme_name": c["scheme_name"], "category": c["category"], "official_url": c["official_url"]})

    if matched:
        return {
            "status": "success",
            "matched_schemes": matched[:4]
        }
    return {
        "status": "no_match",
        "matched_schemes": []
    }

def execute_get_scheme_rules(scheme_id: str) -> Dict[str, Any]:
    cards = _get_knowledge_cards()
    if scheme_id in cards:
        c = cards[scheme_id]
        return {
            "status": "success",
            "scheme_id": c["scheme_id"],
            "scheme_name": c["scheme_name"],
            "eligibility": c["eligibility"],
            "benefits": c["benefits"],
            "official_url": c["official_url"],
            "requires_verification": True
        }
    return {
        "status": "not_found",
        "scheme_id": scheme_id,
        "scheme_name": "Unknown / Unverified Scheme",
        "eligibility": "Scheme not found in verified repository.",
        "benefits": "Unknown.",
        "official_url": "https://myscheme.gov.in/",
        "requires_verification": True
    }

def execute_get_documents(scheme_id: str) -> Dict[str, Any]:
    cards = _get_knowledge_cards()
    if scheme_id in cards:
        c = cards[scheme_id]
        return {
            "status": "success",
            "scheme_id": c["scheme_id"],
            "documents": c["documents"]
        }
    return {
        "status": "not_found",
        "scheme_id": scheme_id,
        "documents": ["Aadhaar Card", "Identity Proof", "Address Proof"]
    }

def execute_find_facility(district: str, facility_type: str = "all") -> Dict[str, Any]:
    district_clean = district.title().strip()
    facilities = [
        {
            "name": f"District Civil Hospital Help Desk, {district_clean}",
            "type": "Public Empanelled Hospital",
            "services": ["Ayushman Bharat PM-JAY Desk", "Golden Card Verification", "Dialysis Unit"],
            "timing": "24x7 Emergency / 09:00 AM - 05:00 PM OPD"
        },
        {
            "name": f"Common Service Center (CSC), {district_clean} HQ",
            "type": "Enrollment & Digital Services Center",
            "services": ["ABHA Card Creation", "e-Shram Registration", "Aadhaar e-KYC"],
            "timing": "09:30 AM - 06:00 PM"
        },
        {
            "name": f"Sub-Divisional Hospital & Wellness Centre, {district_clean}",
            "type": "Community Health Centre",
            "services": ["Maternal & Child Health Care (JSY/JSSK)", "Immunization (Mission Indradhanush)"],
            "timing": "09:00 AM - 04:00 PM"
        }
    ]
    return {
        "status": "success",
        "district": district_clean,
        "facilities": facilities
    }

def execute_create_case_summary(collected_fields: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "success",
        "case_summary": {
            "user_age": collected_fields.get("age"),
            "category": collected_fields.get("category"),
            "primary_need": collected_fields.get("primary_need"),
            "matched_schemes": collected_fields.get("matched_schemes", []),
            "unresolved_questions": collected_fields.get("unresolved_questions", []),
            "urgency_level": collected_fields.get("urgency_level", "STANDARD_VERIFICATION")
        }
    }

def execute_escalate_to_human(case_summary: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "status": "ESCALATED",
        "message": f"Case escalated to designated welfare officer / ASHA coordinator. Reason: {reason}.",
        "handoff_package": {
            "case_summary": case_summary,
            "escalation_reason": reason,
            "handoff_status": "QUEUED_FOR_SUPERVISOR_REVIEW"
        }
    }

def dispatch_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Universal dispatcher for the 7 tools."""
    if tool_name == "emergency_screen":
        return execute_emergency_screen(**arguments)
    elif tool_name == "match_schemes":
        return execute_match_schemes(**arguments)
    elif tool_name == "get_scheme_rules":
        return execute_get_scheme_rules(**arguments)
    elif tool_name == "get_documents":
        return execute_get_documents(**arguments)
    elif tool_name == "find_facility":
        return execute_find_facility(**arguments)
    elif tool_name == "create_case_summary":
        return execute_create_case_summary(**arguments)
    elif tool_name == "escalate_to_human":
        return execute_escalate_to_human(**arguments)
    else:
        raise ValueError(f"Unknown tool name: {tool_name}")
