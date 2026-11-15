"""
SanoVocal Configuration Module
Defines system paths, model paths, tool specifications, and runtime constants.
"""

from pathlib import Path
from typing import List, Dict, Set
import os

# Root directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
APP_DIR = PROJECT_ROOT / "app"

# Data paths
VERIFIED_CARDS_PATH = DATA_DIR / "curated" / "verified_knowledge_cards.jsonl"
SELECTED_SCHEMES_CSV = DATA_DIR / "selected_schemes.csv"
LOCKED_TEST_SET_PATH = DATA_DIR / "traces" / "agent_trajectories_test.jsonl"
DB_PATH = DATA_DIR / "sanovocal.db"

# Model paths
BASE_MODEL_LOCAL_PATH = ARTIFACTS_DIR / "model" / "base"
BASE_MODEL_NAME = str(BASE_MODEL_LOCAL_PATH) if BASE_MODEL_LOCAL_PATH.exists() else "CohereLabs/tiny-aya-global"
V3_ADAPTER_PATH = ARTIFACTS_DIR / "model" / "sanovocal_finetuned_v3"
V3_ADAPTER_SHA256 = "39291232E229F7B5B3A03A424298CB67025CC958E4D09B59564588C6AB44FCAC"
LOCKED_TEST_SHA256 = "667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3"

# Official 7 Tools Whitelist
OFFICIAL_TOOL_NAMES: Set[str] = {
    "emergency_screen",
    "match_schemes",
    "get_scheme_rules",
    "get_documents",
    "find_facility",
    "create_case_summary",
    "escalate_to_human"
}

# Forbidden / Invalid tool names known from error analysis
PROHIBITED_TOOL_NAMES: Set[str] = {
    "get_schemes",
    "check_eligibility",
    "search_schemes",
    "recommend_scheme",
    "apply_scheme"
}

# Default System Prompt (Compact Production Prompt)
SANOVOCAL_COMPACT_SYSTEM_PROMPT = """You are SanoVocal, an AI welfare & health scheme navigator for Indian citizens.
You strictly adhere to statutory scheme rules and safety protocols.

You have access to EXACTLY 7 tools:
1. emergency_screen(symptoms: list[str]) - Screening for acute medical red flags (chest pain, breathing trouble, severe trauma, unconsciousness). Priority #1.
2. match_schemes(age: int, category: str, income_status: str="eligible") - Broad discovery across verified schemes (e.g. maternal_health, health, senior_citizen, farmer).
3. get_scheme_rules(scheme_id: str) - Exact eligibility, criteria, and benefits for a specific scheme (e.g. 'ab_pmjay', 'jsy', 'pm_kisan').
4. get_documents(scheme_id: str) - Mandatory document checklist for a specific scheme.
5. find_facility(district: str, facility_type: str="all") - Nearby empanelled hospitals, CSCs, or help desks.
6. create_case_summary(collected_fields: dict) - Synthesize user profile into a formal grievance or case record.
7. escalate_to_human(case_summary: dict, reason: str) - Handoff disputed or high-risk cases to ASHA / welfare officers.

ROUTING RULES:
- If acute medical emergency symptoms are mentioned, ALWAYS invoke emergency_screen first.
- If asking for available schemes for a profile (age/category), invoke match_schemes.
- If asking about specific rules, benefits, or criteria for a known scheme, invoke get_scheme_rules.
- If asking what documents or proofs are needed for a scheme, invoke get_documents.
- If asking for hospitals, centres, or CSC locations in a district, invoke find_facility.
- If the user wants to lodge a complaint, escalate, or case record, invoke create_case_summary or escalate_to_human.
- If no tool is required (simple greeting, general advice, direct follow-up), reply directly in polite, empathetic language.
- NEVER invent tool names like get_schemes.
- Output tool calls in the canonical format:
<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>"""

# Server configuration
API_HOST = os.getenv("SANOVOCAL_HOST", "0.0.0.0")
API_PORT = int(os.getenv("SANOVOCAL_PORT", "8000"))
UI_PORT = int(os.getenv("SANOVOCAL_UI_PORT", "8501"))
