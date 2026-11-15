"""
SanoVocal Strict Tool Registry & Execution Dispatcher
Guarantees that ONLY the 7 verified tools can ever be executed.
Rejects unauthorized or hallucinated tools like 'get_schemes' with clear feedback.
"""

from typing import Dict, Any, Tuple, Optional
from pydantic import ValidationError
from app.core.config import OFFICIAL_TOOL_NAMES, PROHIBITED_TOOL_NAMES
from app.tools.schemas import TOOL_REGISTRY
from app.tools.engine import dispatch_tool

class ToolExecutionResult:
    def __init__(self, success: bool, tool_name: str, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.success = success
        self.tool_name = tool_name
        self.result = result or {}
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "result": self.result,
            "error": self.error
        }

def validate_and_execute_tool(tool_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
    """
    Validates tool name against whitelist, validates arguments against Pydantic schema,
    and executes via the verified deterministic engine.
    """
    clean_name = tool_name.strip()

    # 1. Check for prohibited/hallucinated tool names
    if clean_name in PROHIBITED_TOOL_NAMES or clean_name not in OFFICIAL_TOOL_NAMES:
        return ToolExecutionResult(
            success=False,
            tool_name=clean_name,
            error=f"Tool '{clean_name}' is not an authorized SanoVocal tool. Authorized tools: {sorted(list(OFFICIAL_TOOL_NAMES))}."
        )

    # 2. Validate input schema with Pydantic
    tool_spec = TOOL_REGISTRY.get(clean_name)
    if not tool_spec:
        return ToolExecutionResult(
            success=False,
            tool_name=clean_name,
            error=f"Tool schema definition missing for '{clean_name}'."
        )

    input_model = tool_spec["input_model"]
    try:
        validated_args = input_model(**arguments).model_dump()
    except ValidationError as ve:
        return ToolExecutionResult(
            success=False,
            tool_name=clean_name,
            error=f"Argument schema validation failed for {clean_name}: {str(ve)}"
        )
    except Exception as e:
        return ToolExecutionResult(
            success=False,
            tool_name=clean_name,
            error=f"Invalid arguments for {clean_name}: {str(e)}"
        )

    # 3. Execute tool deterministically
    try:
        raw_result = dispatch_tool(clean_name, validated_args)
        # Optional: validate output schema
        output_model = tool_spec.get("output_model")
        if output_model:
            output_model(**raw_result)
        return ToolExecutionResult(
            success=True,
            tool_name=clean_name,
            result=raw_result
        )
    except Exception as exec_err:
        return ToolExecutionResult(
            success=False,
            tool_name=clean_name,
            error=f"Execution error in {clean_name}: {str(exec_err)}"
        )
