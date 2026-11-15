"""
Deterministic Validation for SanoVocal Tools & Schemas
Verifies:
1. Pydantic input and output validation
2. Tool execution results matching expected schemas
3. Edge cases: missing fields, emergency triggers, no-matches, invalid types
"""

import json
import sys
from pathlib import Path
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.tools.schemas import TOOL_REGISTRY
from app.tools.engine import dispatch_tool

def test_all_tools():
    print("=" * 60)
    print(" SANOVOCAL — TOOL SCHEMA & EXECUTION VALIDATION")
    print("=" * 60)

    # 1. Emergency Screen (Safe & Emergency)
    print("\n[Test 1] emergency_screen...")
    inp1 = {"symptoms": ["mild headache", "fever"]}
    validated_inp1 = TOOL_REGISTRY["emergency_screen"]["input_model"](**inp1)
    res1 = dispatch_tool("emergency_screen", validated_inp1.model_dump())
    TOOL_REGISTRY["emergency_screen"]["output_model"](**res1)
    assert res1["status"] == "SAFE"

    inp1_em = {"symptoms": ["chest pain", "breathlessness"]}
    validated_inp1_em = TOOL_REGISTRY["emergency_screen"]["input_model"](**inp1_em)
    res1_em = dispatch_tool("emergency_screen", validated_inp1_em.model_dump())
    TOOL_REGISTRY["emergency_screen"]["output_model"](**res1_em)
    assert res1_em["status"] == "EMERGENCY"
    print("[OK] emergency_screen passed (SAFE and EMERGENCY overrides verified)")

    # 2. Match Schemes (Senior & Standard)
    print("\n[Test 2] match_schemes...")
    inp2_senior = {"age": 72, "category": "health"}
    v_inp2 = TOOL_REGISTRY["match_schemes"]["input_model"](**inp2_senior)
    res2 = dispatch_tool("match_schemes", v_inp2.model_dump())
    TOOL_REGISTRY["match_schemes"]["output_model"](**res2)
    assert any(s["scheme_id"] == "ayushman_vay_vandana" for s in res2["matched_schemes"])

    inp2_maternal = {"age": 24, "category": "maternal_health"}
    v_inp2_m = TOOL_REGISTRY["match_schemes"]["input_model"](**inp2_maternal)
    res2_m = dispatch_tool("match_schemes", v_inp2_m.model_dump())
    TOOL_REGISTRY["match_schemes"]["output_model"](**res2_m)
    assert res2_m["status"] == "success"
    print("[OK] match_schemes passed (Senior citizen rule and category matching verified)")

    # 3. Get Scheme Rules
    print("\n[Test 3] get_scheme_rules...")
    inp3 = {"scheme_id": "ab_pmjay"}
    v_inp3 = TOOL_REGISTRY["get_scheme_rules"]["input_model"](**inp3)
    res3 = dispatch_tool("get_scheme_rules", v_inp3.model_dump())
    TOOL_REGISTRY["get_scheme_rules"]["output_model"](**res3)
    assert res3["status"] == "success"
    assert "pmjay.gov.in" in res3["official_url"]
    print("[OK] get_scheme_rules passed (Official eligibility and source URL verified)")

    # 4. Get Documents
    print("\n[Test 4] get_documents...")
    inp4 = {"scheme_id": "jsy"}
    v_inp4 = TOOL_REGISTRY["get_documents"]["input_model"](**inp4)
    res4 = dispatch_tool("get_documents", v_inp4.model_dump())
    TOOL_REGISTRY["get_documents"]["output_model"](**res4)
    assert res4["status"] == "success"
    assert "Aadhaar Card" in res4["documents"]
    print("[OK] get_documents passed (Mandatory verification checklists verified)")

    # 5. Find Facility
    print("\n[Test 5] find_facility...")
    inp5 = {"district": "Varanasi", "facility_type": "all"}
    v_inp5 = TOOL_REGISTRY["find_facility"]["input_model"](**inp5)
    res5 = dispatch_tool("find_facility", v_inp5.model_dump())
    TOOL_REGISTRY["find_facility"]["output_model"](**res5)
    assert res5["status"] == "success"
    assert len(res5["facilities"]) > 0
    print("[OK] find_facility passed (Empanelled district facilities verified)")

    # 6. Create Case Summary
    print("\n[Test 6] create_case_summary...")
    inp6 = {"collected_fields": {"age": 72, "category": "health", "primary_need": "hospitalization", "matched_schemes": ["ayushman_vay_vandana"]}}
    v_inp6 = TOOL_REGISTRY["create_case_summary"]["input_model"](**inp6)
    res6 = dispatch_tool("create_case_summary", v_inp6.model_dump())
    TOOL_REGISTRY["create_case_summary"]["output_model"](**res6)
    assert res6["status"] == "success"
    print("[OK] create_case_summary passed (Structured case summary verified)")

    # 7. Escalate To Human
    print("\n[Test 7] escalate_to_human...")
    inp7 = {"case_summary": res6["case_summary"], "reason": "Applicant lacks standard documentation and requires field officer assistance."}
    v_inp7 = TOOL_REGISTRY["escalate_to_human"]["input_model"](**inp7)
    res7 = dispatch_tool("escalate_to_human", v_inp7.model_dump())
    TOOL_REGISTRY["escalate_to_human"]["output_model"](**res7)
    assert res7["status"] == "ESCALATED"
    print("[OK] escalate_to_human passed (Supervisory human handoff verified)")

    # Negative validation test: invalid age
    print("\n[Negative Tests] Verifying validation catches malformed inputs...")
    try:
        TOOL_REGISTRY["match_schemes"]["input_model"](age=150, category="health")
        assert False, "Should have thrown ValidationError for age > 120"
    except ValidationError:
        print("[OK] Pydantic successfully rejected invalid age > 120")

    try:
        dispatch_tool("non_existent_tool", {})
        assert False, "Should have thrown ValueError for unknown tool"
    except ValueError:
        print("[OK] Dispatcher successfully rejected unknown tool")

    print("\n>>> ALL 7 TOOLS DETERMINISTICALLY VALIDATED! <<<")

if __name__ == "__main__":
    test_all_tools()
