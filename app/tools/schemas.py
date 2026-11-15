"""
SanoVocal Tool Schemas
Strict Pydantic models and JSON Schema specifications for SanoVocal's 7 core agent tools.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# 1. Emergency Screening Tool
# ---------------------------------------------------------------------------

class EmergencyScreenInput(BaseModel):
    symptoms: List[str] = Field(
        ...,
        description="List of physical symptoms reported by the user to check for emergency red-flags."
    )

class EmergencyScreenOutput(BaseModel):
    status: str = Field(..., description="'EMERGENCY' or 'SAFE'")
    action: str = Field(..., description="Action required by the navigator")
    emergency_code: Optional[str] = Field(None, description="Triage emergency code if applicable")

# ---------------------------------------------------------------------------
# 2. Scheme Matching Tool
# ---------------------------------------------------------------------------

class MatchSchemesInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age of the citizen/applicant in years")
    category: str = Field(..., min_length=2, description="Target category or domain (e.g. health, maternal_health, housing, agriculture, etc.)")
    income_status: Optional[str] = Field("eligible", description="Income qualification descriptor, e.g. BPL, rural, low_income, eligible")

class MatchSchemesOutput(BaseModel):
    status: str = Field(..., description="'success' or 'no_match'")
    matched_schemes: List[Dict[str, Any]] = Field(default_factory=list, description="List of matched scheme summary objects")

# ---------------------------------------------------------------------------
# 3. Scheme Rules & Eligibility Tool
# ---------------------------------------------------------------------------

class GetSchemeRulesInput(BaseModel):
    scheme_id: str = Field(..., min_length=2, description="Unique identifier for the scheme (e.g. 'ab_pmjay')")

class GetSchemeRulesOutput(BaseModel):
    status: str = Field(..., description="'success' or 'not_found'")
    scheme_id: str
    scheme_name: str
    eligibility: str
    benefits: str
    official_url: str
    requires_verification: bool = True

# ---------------------------------------------------------------------------
# 4. Document Checklist Tool
# ---------------------------------------------------------------------------

class GetDocumentsInput(BaseModel):
    scheme_id: str = Field(..., min_length=2, description="Unique identifier for the scheme")

class GetDocumentsOutput(BaseModel):
    status: str = Field(..., description="'success' or 'not_found'")
    scheme_id: str
    documents: List[str]

# ---------------------------------------------------------------------------
# 5. Help Center / Facility Lookup Tool
# ---------------------------------------------------------------------------

class FindFacilityInput(BaseModel):
    district: str = Field(..., min_length=2, description="District or city name for facility search")
    facility_type: Optional[str] = Field("all", description="Type of facility: hospital, CSC, help_desk, all")

class FindFacilityOutput(BaseModel):
    status: str = Field(..., description="'success' or 'no_facilities'")
    district: str
    facilities: List[Dict[str, Any]]

# ---------------------------------------------------------------------------
# 6. Case Summary Tool
# ---------------------------------------------------------------------------

class CreateCaseSummaryInput(BaseModel):
    collected_fields: Dict[str, Any] = Field(..., description="Key user attributes collected across conversation")

class CreateCaseSummaryOutput(BaseModel):
    status: str = Field(..., description="'success' or 'error'")
    case_summary: Dict[str, Any]

# ---------------------------------------------------------------------------
# 7. Human Escalation Tool
# ---------------------------------------------------------------------------

class EscalateToHumanInput(BaseModel):
    case_summary: Dict[str, Any] = Field(..., description="Structured summary of the user's situation")
    reason: str = Field(..., min_length=3, description="Justification for human handoff")

class EscalateToHumanOutput(BaseModel):
    status: str = Field(..., description="'ESCALATED'")
    message: str
    handoff_package: Dict[str, Any]

# ---------------------------------------------------------------------------
# Tool Registry Map
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "emergency_screen": {
        "input_model": EmergencyScreenInput,
        "output_model": EmergencyScreenOutput,
        "description": "Screens user utterances for acute medical emergencies. If detected, immediate triage override occurs."
    },
    "match_schemes": {
        "input_model": MatchSchemesInput,
        "output_model": MatchSchemesOutput,
        "description": "Matches citizen profile (age, category, income) against the verified 40-scheme whitelist."
    },
    "get_scheme_rules": {
        "input_model": GetSchemeRulesInput,
        "output_model": GetSchemeRulesOutput,
        "description": "Retrieves official statutory eligibility guidelines and benefits for a verified scheme ID."
    },
    "get_documents": {
        "input_model": GetDocumentsInput,
        "output_model": GetDocumentsOutput,
        "description": "Retrieves the mandatory document verification checklist for a specific scheme."
    },
    "find_facility": {
        "input_model": FindFacilityInput,
        "output_model": FindFacilityOutput,
        "description": "Discovers nearby empanelled healthcare facilities, CSCs, or government welfare help desks."
    },
    "create_case_summary": {
        "input_model": CreateCaseSummaryInput,
        "output_model": CreateCaseSummaryOutput,
        "description": "Compiles collected eligibility criteria into a structured case payload for records or handoff."
    },
    "escalate_to_human": {
        "input_model": EscalateToHumanInput,
        "output_model": EscalateToHumanOutput,
        "description": "Transfers complex, contested, high-risk, or unresolved cases to a human caseworker or ASHA worker."
    }
}
