"""
SanoVocal Chat & Workflow API Routes
"""

import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.session import SessionContext
from app.model.inference import generate_assistant_response
from app.storage.repository import DataRepository

router = APIRouter(prefix="/v1/chat", tags=["Chat"])

# Active in-memory session registry (backed by persistent SQLite)
SESSION_STORE: Dict[str, SessionContext] = {}

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Citizen's voice transcription or text query")
    session_id: Optional[str] = Field(None, description="Existing session identifier")
    language: Optional[str] = Field("en", description="Citizen's language ('en', 'hi', 'hinglish')")

class ChatResponse(BaseModel):
    session_id: str
    response: str
    safety_verdict: Dict[str, Any]
    tool_call: Optional[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    latency_ms: int
    bypassed_llm: bool

class CaseCreateRequest(BaseModel):
    session_id: str
    user_name: Optional[str] = None
    age: Optional[int] = None
    district: Optional[str] = None
    category: Optional[str] = None
    matched_schemes: Optional[List[str]] = None
    escalation_reason: Optional[str] = None
    urgency_level: str = "STANDARD"

class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = ""

@router.post("/message", response_model=ChatResponse)
def post_chat_message(req: ChatRequest):
    sess_id = req.session_id or str(uuid.uuid4())
    if sess_id not in SESSION_STORE:
        session = SessionContext(session_id=sess_id, language=req.language or "en")
        SESSION_STORE[sess_id] = session
        DataRepository.save_session(sess_id, session.language)
    else:
        session = SESSION_STORE[sess_id]
        if req.language:
            session.language = req.language

    # Run inference orchestrator
    result = generate_assistant_response(
        user_text=req.message,
        session=session
    )

    # Persist in SQLite
    user_msg_id = str(uuid.uuid4())
    asst_msg_id = str(uuid.uuid4())
    DataRepository.log_message(user_msg_id, sess_id, "user", req.message)
    DataRepository.log_message(
        asst_msg_id,
        sess_id,
        "assistant",
        result["response"],
        tool_calls=[result["tool_call"]] if result["tool_call"] else None,
        tool_results=[result["tool_result"]] if result["tool_result"] else None
    )

    if result["tool_call"]:
        DataRepository.log_tool_event(
            session_id=sess_id,
            tool_name=result["tool_call"]["name"],
            arguments=result["tool_call"]["arguments"],
            result=result["tool_result"] or {},
            status="SUCCESS" if result["tool_call"].get("success") else "FAILED"
        )

    return ChatResponse(
        session_id=sess_id,
        response=result["response"],
        safety_verdict=result["safety_verdict"],
        tool_call=result["tool_call"],
        tool_result=result["tool_result"],
        latency_ms=result["latency_ms"],
        bypassed_llm=result["bypassed_llm"]
    )

@router.get("/cases")
def list_cases(limit: int = 50):
    return DataRepository.get_cases(limit=limit)

@router.post("/cases")
def create_case(req: CaseCreateRequest):
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    case_record = DataRepository.create_case(
        case_id=case_id,
        session_id=req.session_id,
        user_name=req.user_name,
        age=req.age,
        district=req.district,
        category=req.category,
        matched_schemes=req.matched_schemes,
        escalation_reason=req.escalation_reason,
        urgency_level=req.urgency_level
    )
    return case_record

@router.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    DataRepository.add_feedback(req.session_id, req.rating, req.comments or "")
    return {"status": "success", "message": "Feedback recorded."}

@router.get("/metrics")
def get_metrics():
    return DataRepository.get_dashboard_metrics()
