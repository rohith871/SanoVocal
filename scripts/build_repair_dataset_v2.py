"""
SanoVocal Tool Contract Repair Dataset V2 Generator
Builds a high-precision dataset teaching semantic tool routing, contrastive discrimination,
exact official tool contracts, multilingual parity, and no-tool decisions.
"""

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.tools.engine import dispatch_tool, _get_knowledge_cards
from app.tools.schemas import TOOL_REGISTRY

SANOVOCAL_SYSTEM_PROMPT = """You are SanoVocal, an AI assistant for navigating Indian health and social welfare schemes.

### Official Tools
You have access to EXACTLY SEVEN tools. You must format tool calls using:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>

1. emergency_screen:
   - Purpose: Screen physical symptoms for clinical red flags.
   - Arguments: symptoms (list of strings).
   - Call when: User mentions chest pain, stroke symptoms, severe breathing difficulty, massive bleeding, or other emergencies.
   - Do NOT call when: User is merely asking general scheme rules, document checklists, or has no symptoms.

2. match_schemes:
   - Purpose: Find eligible welfare schemes based on age, category, income.
   - Arguments: age (integer), category (string: health, maternal_health, agriculture, housing, etc.), income_status (optional string).
   - Call when: User seeks scheme recommendations and provides profile details.
   - Do NOT call when: Scheme is already known or user has acute medical emergency.

3. get_scheme_rules:
   - Purpose: Fetch verified rules, eligibility, benefits, and official portal URL for a specific scheme.
   - Arguments: scheme_id (string, e.g. 'ab_pmjay', 'jsy', 'pm_kisan').
   - Call when: User asks about benefits, eligibility, or rules of a specific identified scheme.
   - Do NOT call when: User needs general exploration across multiple schemes.

4. get_documents:
   - Purpose: Retrieve required document checklist for application.
   - Arguments: scheme_id (string).
   - Call when: User asks what papers, ID, or documents are needed for a scheme.
   - Do NOT call when: User asks about general eligibility or emergency care.

5. find_facility:
   - Purpose: Find empanelled hospitals, common service centers (CSCs), or help desks.
   - Arguments: district (string), facility_type (optional string: hospital, CSC, help_desk, all).
   - Call when: User asks where to go or seeks nearest center in a district.
   - Do NOT call when: User is asking for rules or documents.

6. create_case_summary:
   - Purpose: Compile structured user profile summary before application or handoff.
   - Arguments: collected_fields (dictionary of collected profile data).
   - Call when: Sufficient profile information has been gathered and case needs structuring.
   - Do NOT call when: Crucial initial questions remain unanswered.

7. escalate_to_human:
   - Purpose: Hand off complex disputes, discrimination, or hospital denial of care to a human officer.
   - Arguments: case_summary (dictionary), reason (string).
   - Call when: Hospital denies cashless treatment, fraud/bribery occurs, or automated resolution fails.
   - Do NOT call when: Issue can be resolved through standard scheme navigation.

### SanoVocal Safety & Operating Rules
- Never diagnose medical conditions.
- Never prescribe medication or recommend dosage.
- Never invent scheme names, rules, or official URLs.
- Never invent tool names; use ONLY the seven tools defined above.
- If emergency symptoms are present, prioritize emergency triage over scheme search.
- If a tool is required, emit the <tool_call> block immediately.
- If no tool is needed, respond helpfully and clearly."""

OFFICIAL_TOOLS = {
    "emergency_screen", "match_schemes", "get_scheme_rules",
    "get_documents", "find_facility", "create_case_summary", "escalate_to_human"
}

def load_test_tasks() -> set:
    test_path = ROOT / "data" / "traces" / "agent_trajectories_test.jsonl"
    with open(test_path, "r", encoding="utf-8") as f:
        return set(json.loads(line).get("task", "").strip().lower() for line in f if line.strip())

def make_record(
    rec_id: str,
    category: str,
    task: str,
    language: str,
    tool_name: str | None,
    args: Dict[str, Any] | None,
    thought: str,
    final_response: str,
    metadata: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Build a validated V2 record with system prompt and verified execution."""
    if metadata is None:
        metadata = {}
    metadata.update({"category": category, "language": language})

    steps = []
    messages = [{"role": "system", "content": SANOVOCAL_SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": task})

    if tool_name:
        if tool_name not in OFFICIAL_TOOLS:
            raise ValueError(f"CRITICAL: Attempted to add invalid tool: {tool_name}")
        
        # Validate args against registry Pydantic model
        validated_args = TOOL_REGISTRY[tool_name]["input_model"](**args).model_dump()
        obs = dispatch_tool(tool_name, validated_args)
        obs_str = json.dumps(obs, ensure_ascii=False)

        steps.append({
            "step_number": 1,
            "state": "ACTIVE",
            "action": {
                "name": tool_name,
                "arguments": validated_args
            },
            "observation": obs_str,
            "thought": thought,
            "reward": 1.0
        })

        tc_block = f"<tool_call>\n{{\"name\": \"{tool_name}\", \"arguments\": {json.dumps(validated_args, ensure_ascii=False)}}}\n</tool_call>"
        assistant_turn = f"{thought}\n{tc_block}" if thought else tc_block

        messages.append({"role": "assistant", "content": assistant_turn})
        messages.append({"role": "tool", "content": obs_str})
        messages.append({"role": "assistant", "content": final_response})
    else:
        # NO_TOOL scenario
        messages.append({"role": "assistant", "content": final_response})

    return {
        "id": rec_id,
        "category": category,
        "task": task,
        "language": language,
        "steps": steps,
        "final_response": final_response,
        "messages": messages,
        "metadata": metadata
    }
