"""
SanoVocal Dual-Format Output Parser
Extracts tool calls and natural language responses from model generations.
Supports both canonical <tool_call> JSON tags and fallback pseudo-markup.
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple

class ParsedGeneration:
    def __init__(self, raw_text: str, tool_calls: List[Dict[str, Any]], clean_response: str):
        self.raw_text = raw_text
        self.tool_calls = tool_calls  # List of {"name": str, "arguments": dict}
        self.clean_response = clean_response.strip()

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_tool_calls": self.has_tool_calls,
            "tool_calls": self.tool_calls,
            "clean_response": self.clean_response
        }

def parse_model_generation(generation_text: str) -> ParsedGeneration:
    """
    Parses generation text to identify tool calls and extract clean text.
    Handles:
    1. Canonical: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. Fallback: Tool call: tool_name(arg1=val1, ...)
    3. Fallback: Tool call: {"name": "...", ...}
    """
    text = generation_text.strip()
    tool_calls = []

    # 1. Canonical <tool_call>...</tool_call> extraction
    canonical_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    canonical_matches = list(canonical_pattern.finditer(text))

    cleaned_text = text

    if canonical_matches:
        for m in canonical_matches:
            body = m.group(1).strip()
            try:
                data = json.loads(body)
                if isinstance(data, dict) and "name" in data:
                    args = data.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    tool_calls.append({
                        "name": str(data["name"]).strip(),
                        "arguments": args if isinstance(args, dict) else {}
                    })
            except Exception:
                # If strict JSON decode fails, attempt to extract via regex
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', body)
                if name_match:
                    tool_name = name_match.group(1).strip()
                    tool_calls.append({
                        "name": tool_name,
                        "arguments": {}
                    })
        cleaned_text = canonical_pattern.sub("", text).strip()

    # 2. If no canonical tool call found, check for pseudo-markup fallback:
    # Example: "Tool call: match_schemes(age=65, category='health')" or "Tool call: match_schemes{...}"
    if not tool_calls:
        pseudo_match = re.search(r'Tool call:\s*([a-zA-Z0-9_]+)\s*\((.*?)\)', text, re.DOTALL)
        if pseudo_match:
            tool_name = pseudo_match.group(1).strip()
            args_str = pseudo_match.group(2).strip()
            arguments = {}
            if args_str:
                # Parse key=value pairs: e.g. age=65, category="health", income_status="BPL"
                arg_pairs = re.findall(r'([a-zA-Z0-9_]+)\s*=\s*(["\']?)(.*?)\2(?=,\s*[a-zA-Z0-9_]+\s*=|\s*$)', args_str)
                for k, _, v in arg_pairs:
                    val = v.strip().strip("'\"")
                    if val.isdigit():
                        arguments[k] = int(val)
                    elif val.lower() == "true":
                        arguments[k] = True
                    elif val.lower() == "false":
                        arguments[k] = False
                    else:
                        arguments[k] = val
            tool_calls.append({
                "name": tool_name,
                "arguments": arguments
            })
            cleaned_text = re.sub(r'Tool call:\s*[a-zA-Z0-9_]+\s*\(.*?\)', '', text, flags=re.DOTALL).strip()
        else:
            # Fallback 2: "Tool call: {"name": ...}"
            json_pseudo = re.search(r'Tool call:\s*(\{.*?\})', text, re.DOTALL)
            if json_pseudo:
                try:
                    data = json.loads(json_pseudo.group(1))
                    if isinstance(data, dict) and "name" in data:
                        tool_calls.append({
                            "name": str(data["name"]).strip(),
                            "arguments": data.get("arguments", {})
                        })
                        cleaned_text = text[:json_pseudo.start()] + text[json_pseudo.end():]
                        cleaned_text = cleaned_text.strip()
                except Exception:
                    pass

    return ParsedGeneration(
        raw_text=generation_text,
        tool_calls=tool_calls,
        clean_response=cleaned_text
    )
