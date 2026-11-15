"""
SanoVocal Session & Conversation State Management
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class MessageRecord(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str  # 'user', 'assistant', 'system', 'tool'
    text: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class UserProfile(BaseModel):
    age: Optional[int] = None
    category: Optional[str] = None
    income_status: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    disability_status: Optional[bool] = None
    gender: Optional[str] = None

class SessionContext(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    language: str = "en"  # 'en', 'hi', 'hinglish'
    profile: UserProfile = Field(default_factory=UserProfile)
    history: List[MessageRecord] = Field(default_factory=list)
    active_case: Optional[Dict[str, Any]] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_message(self, sender: str, text: str, tool_calls: Optional[List[Dict[str, Any]]] = None, tool_results: Optional[List[Dict[str, Any]]] = None) -> MessageRecord:
        msg = MessageRecord(
            sender=sender,
            text=text,
            tool_calls=tool_calls,
            tool_results=tool_results
        )
        self.history.append(msg)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return msg

    def update_profile(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.profile, k) and v is not None:
                setattr(self.profile, k, v)
        self.updated_at = datetime.now(timezone.utc).isoformat()
