"""
SanoVocal End-to-End Inference Orchestrator
Connects Deterministic Safety, LLM Generation, Parsing, Tool Execution, and Evidence Formulation.
"""

import time
import json
from typing import Dict, Any, List, Optional
import torch

from app.core.config import SANOVOCAL_COMPACT_SYSTEM_PROMPT
from app.core.safety import evaluate_deterministic_safety
from app.core.session import SessionContext
from app.model.loader import load_v3_model
from app.model.parser import parse_model_generation
from app.tools.registry import validate_and_execute_tool

def format_system_prompt_chat(system_prompt: str, history: List[Dict[str, str]], user_text: str) -> str:
    """
    Constructs the prompt for Tiny Aya Global model.
    """
    prompt = f"<|start_of_turn|>system\n{system_prompt}<|end_of_turn|>\n"
    for turn in history[-6:]:  # Keep recent history
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            prompt += f"<|start_of_turn|>user\n{content}<|end_of_turn|>\n"
        elif role == "assistant":
            prompt += f"<|start_of_turn|>assistant\n{content}<|end_of_turn|>\n"
    prompt += f"<|start_of_turn|>user\n{user_text}<|end_of_turn|>\n<|start_of_turn|>assistant\n"
    return prompt

def generate_assistant_response(
    user_text: str,
    session: Optional[SessionContext] = None,
    temperature: float = 0.1,
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    """
    Orchestrates complete SanoVocal response pipeline for a user message.
    """
    start_time = time.time()
    if session is None:
        session = SessionContext()

    # Step 1: Deterministic Safety Check
    safety_verdict = evaluate_deterministic_safety(user_text)
    if not safety_verdict.is_safe:
        override_text = safety_verdict.override_response
        session.add_message("user", user_text)
        session.add_message("assistant", override_text)
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "response": override_text,
            "safety_verdict": safety_verdict.to_dict(),
            "tool_call": None,
            "tool_result": None,
            "latency_ms": latency_ms,
            "session_id": session.session_id,
            "bypassed_llm": True
        }

    # Step 2: Prepare Chat History
    chat_history = []
    for msg in session.history:
        chat_history.append({"role": msg.sender, "content": msg.text})

    prompt = format_system_prompt_chat(SANOVOCAL_COMPACT_SYSTEM_PROMPT, chat_history, user_text)

    # Step 3: LLM Inference via Singleton Loader
    tokenizer, model = load_v3_model()
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to model's device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # Slice out generated tokens only
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = output_ids[0][input_length:]
    raw_generation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Step 4: Parse Generation
    parsed = parse_model_generation(raw_generation)

    executed_tool = None
    tool_output = None
    final_response = parsed.clean_response

    # Step 5: Execute Tool if detected
    if parsed.has_tool_calls:
        target_tool = parsed.tool_calls[0]
        tool_name = target_tool["name"]
        tool_args = target_tool["arguments"]

        # Validate and Execute Tool strictly
        tool_exec = validate_and_execute_tool(tool_name, tool_args)
        executed_tool = {
            "name": tool_name,
            "arguments": tool_args,
            "success": tool_exec.success,
            "error": tool_exec.error
        }
        tool_output = tool_exec.result

        # Synthesize final response if empty or enrich with ground truth
        if tool_exec.success:
            if tool_name == "emergency_screen":
                if tool_output.get("status") == "EMERGENCY":
                    final_response = (
                        "🚨 **EMERGENCY MEDICAL ALERT / आपातकालीन चेतावनी**\n\n"
                        f"{tool_output.get('action')}\n\n"
                        "- Please call **112** or **108** immediately."
                    )
                else:
                    final_response = "Checked for acute emergency red-flags: No immediate medical emergency detected. How can I assist you with government health or welfare schemes?"
            elif tool_name == "match_schemes":
                matched = tool_output.get("matched_schemes", [])
                if matched:
                    lines = ["Here are the official schemes matching your profile:\n"]
                    for s in matched:
                        lines.append(f"- **{s.get('scheme_name')}** ({s.get('scheme_id')}): Verified under {s.get('category')}. [Official Portal]({s.get('official_url')})")
                    final_response = "\n".join(lines)
                else:
                    final_response = "I searched the verified national welfare repository, but could not find an exact match for the specified criteria. Please verify your age or category."
            elif tool_name == "get_scheme_rules":
                if tool_output.get("status") == "success":
                    final_response = (
                        f"### {tool_output.get('scheme_name')} ({tool_output.get('scheme_id')})\n\n"
                        f"**Eligibility Guidelines:**\n{tool_output.get('eligibility')}\n\n"
                        f"**Benefits & Coverage:**\n{tool_output.get('benefits')}\n\n"
                        f"Official portal: {tool_output.get('official_url')}"
                    )
                else:
                    final_response = f"Scheme '{tool_args.get('scheme_id')}' was not found in the verified repository."
            elif tool_name == "get_documents":
                docs = tool_output.get("documents", [])
                lines = [f"### Mandatory Verification Documents ({tool_args.get('scheme_id')}):\n"]
                for d in docs:
                    lines.append(f"- [x] {d}")
                final_response = "\n".join(lines)
            elif tool_name == "find_facility":
                facs = tool_output.get("facilities", [])
                lines = [f"### Empanelled Facilities & Help Desks in {tool_output.get('district')}:\n"]
                for f in facs:
                    lines.append(f"- **{f.get('name')}** ({f.get('type')})\n  Services: {', '.join(f.get('services', []))}\n  Timing: {f.get('timing')}")
                final_response = "\n".join(lines)
            elif tool_name == "create_case_summary":
                summary = tool_output.get("case_summary", {})
                final_response = (
                    "### Case Summary Record Created\n\n"
                    f"A structured case profile has been created:\n"
                    f"- Age: {summary.get('user_age', 'Not specified')}\n"
                    f"- Category: {summary.get('category', 'Not specified')}\n"
                    f"- Need: {summary.get('primary_need', 'General Navigation')}\n"
                    f"- Status: {summary.get('urgency_level')}"
                )
            elif tool_name == "escalate_to_human":
                final_response = (
                    "### Human Escalation Initiated / मानव सहायता\n\n"
                    f"{tool_output.get('message')}\n\n"
                    "Your case packet has been transferred to the local ASHA supervisor / grievance officer."
                )
        else:
            final_response = f"I attempted to execute `{tool_name}`, but encountered an issue: {tool_exec.error}."

    if not final_response:
        final_response = "I am ready to help you navigate Indian welfare and healthcare schemes. Please specify your age, category, or the scheme you wish to explore."

    # Update session history
    session.add_message("user", user_text)
    session.add_message(
        "assistant",
        final_response,
        tool_calls=[executed_tool] if executed_tool else None,
        tool_results=[tool_output] if tool_output else None
    )

    latency_ms = int((time.time() - start_time) * 1000)
    return {
        "response": final_response,
        "safety_verdict": safety_verdict.to_dict(),
        "tool_call": executed_tool,
        "tool_result": tool_output,
        "raw_generation": raw_generation,
        "latency_ms": latency_ms,
        "session_id": session.session_id,
        "bypassed_llm": False
    }
