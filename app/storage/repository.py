"""
SanoVocal Data Repository
High-level data access for sessions, messages, tool logs, case records, and analytics.
"""

import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from app.storage.database import get_db_connection

class DataRepository:
    @staticmethod
    def save_session(session_id: str, language: str = "en", metadata: Optional[Dict[str, Any]] = None):
        now = datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata or {})
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, language, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    language = excluded.language,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
            """, (session_id, language, now, now, meta_str))
            conn.commit()

    @staticmethod
    def log_message(message_id: str, session_id: str, sender: str, text: str,
                    tool_calls: Optional[List[Dict[str, Any]]] = None,
                    tool_results: Optional[List[Dict[str, Any]]] = None):
        now = datetime.now(timezone.utc).isoformat()
        tc_str = json.dumps(tool_calls) if tool_calls else None
        tr_str = json.dumps(tool_results) if tool_results else None
        with get_db_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO messages (message_id, session_id, sender, text, tool_calls, tool_results, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (message_id, session_id, sender, text, tc_str, tr_str, now))
            conn.commit()

    @staticmethod
    def log_tool_event(session_id: str, tool_name: str, arguments: Dict[str, Any], result: Dict[str, Any], status: str):
        now = datetime.now(timezone.utc).isoformat()
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO tool_events (session_id, tool_name, arguments, result, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, tool_name, json.dumps(arguments), json.dumps(result), status, now))
            conn.commit()

    @staticmethod
    def create_case(case_id: str, session_id: str, user_name: Optional[str], age: Optional[int],
                    district: Optional[str], category: Optional[str],
                    matched_schemes: Optional[List[str]] = None,
                    escalation_reason: Optional[str] = None,
                    urgency_level: str = "STANDARD") -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        schemes_str = json.dumps(matched_schemes or [])
        with get_db_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cases (case_id, session_id, user_name, age, district, category, matched_schemes, escalation_reason, urgency_level, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (case_id, session_id, user_name, age, district, category, schemes_str, escalation_reason, urgency_level, now))
            conn.commit()
        return {
            "case_id": case_id,
            "session_id": session_id,
            "user_name": user_name,
            "age": age,
            "district": district,
            "category": category,
            "matched_schemes": matched_schemes or [],
            "escalation_reason": escalation_reason,
            "urgency_level": urgency_level,
            "created_at": now
        }

    @staticmethod
    def get_cases(limit: int = 50) -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM cases ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def add_feedback(session_id: str, rating: int, comments: str = ""):
        now = datetime.now(timezone.utc).isoformat()
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO feedback (session_id, rating, comments, timestamp)
                VALUES (?, ?, ?, ?)
            """, (session_id, rating, comments, now))
            conn.commit()

    @staticmethod
    def get_dashboard_metrics() -> Dict[str, Any]:
        with get_db_connection() as conn:
            total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            total_tool_calls = conn.execute("SELECT COUNT(*) FROM tool_events").fetchone()[0]
            total_cases = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]

            tool_breakdown = {}
            for row in conn.execute("SELECT tool_name, COUNT(*) as cnt FROM tool_events GROUP BY tool_name"):
                tool_breakdown[row[0]] = row[1]

            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "total_tool_calls": total_tool_calls,
                "total_cases": total_cases,
                "tool_breakdown": tool_breakdown
            }
