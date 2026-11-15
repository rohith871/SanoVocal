"""
SanoVocal Production Test Suite
Verifies Scenarios A through J, System Integrity, Storage, Safety, and Voice Pipelines.
"""

import hashlib
import unittest
from pathlib import Path

from app.core.config import (
    OFFICIAL_TOOL_NAMES, PROHIBITED_TOOL_NAMES,
    V3_ADAPTER_PATH, V3_ADAPTER_SHA256,
    LOCKED_TEST_SET_PATH, LOCKED_TEST_SHA256
)
from app.core.safety import evaluate_deterministic_safety
from app.core.session import SessionContext
from app.tools.registry import validate_and_execute_tool
from app.tools.engine import (
    execute_emergency_screen, execute_match_schemes,
    execute_get_scheme_rules, execute_get_documents,
    execute_find_facility, execute_create_case_summary,
    execute_escalate_to_human
)
from app.model.parser import parse_model_generation
from app.storage.repository import DataRepository
from app.voice.tts import synthesize_speech
from app.voice.stt import transcribe_audio_bytes


class TestSanoVocalProduct(unittest.TestCase):

    def test_system_integrity_hashes(self):
        """Verifies cryptographic hashes of locked test set and frozen V3 adapter."""
        # 1. Locked test set hash check
        self.assertTrue(LOCKED_TEST_SET_PATH.exists(), "Locked test set file must exist.")
        with open(LOCKED_TEST_SET_PATH, "rb") as f:
            test_hash = hashlib.sha256(f.read()).hexdigest().upper()
        self.assertEqual(test_hash, LOCKED_TEST_SHA256, "Locked test set SHA256 must remain strictly unchanged.")

        # 2. V3 adapter hash check
        adapter_bin = V3_ADAPTER_PATH / "adapter_model.safetensors"
        self.assertTrue(adapter_bin.exists(), "V3 adapter safetensors must exist.")
        with open(adapter_bin, "rb") as f:
            adapter_hash = hashlib.sha256(f.read()).hexdigest().upper()
        self.assertEqual(adapter_hash, V3_ADAPTER_SHA256, "V3 adapter SHA256 must match frozen record.")

    def test_scenario_a_emergency_screening(self):
        """Scenario A: Emergency detection and triage override."""
        # Deterministic safety triage check
        emergency_inputs = [
            "My father has severe chest pain and breathlessness!",
            "Patient is unconscious and bleeding heavily",
            "Mummy ko saas lene me dikkat aur chhati me dard hai"
        ]
        for query in emergency_inputs:
            verdict = evaluate_deterministic_safety(query)
            self.assertFalse(verdict.is_safe)
            self.assertEqual(verdict.category, "EMERGENCY_OVERRIDE")
            self.assertIn("112", verdict.override_response)

        # Tool execution check
        res_emerg = execute_emergency_screen(["chest pain", "breathlessness"])
        self.assertEqual(res_emerg["status"], "EMERGENCY")
        self.assertEqual(res_emerg["emergency_code"], "RED_FLAG_TRIAGE")

        res_safe = execute_emergency_screen(["mild cough", "general inquiry"])
        self.assertEqual(res_safe["status"], "SAFE")

    def test_scenario_b_scheme_matching(self):
        """Scenario B: Scheme Search for eligible citizens."""
        # Match for senior citizen health
        res = validate_and_execute_tool("match_schemes", {"age": 72, "category": "health", "income_status": "eligible"})
        self.assertTrue(res.success)
        self.assertEqual(res.result.get("status"), "success")
        matched = res.result.get("matched_schemes", [])
        self.assertGreaterEqual(len(matched), 1)
        scheme_ids = [s["scheme_id"] for s in matched]
        self.assertIn("ayushman_vay_vandana", scheme_ids)

        # Match for maternal health
        res_mat = validate_and_execute_tool("match_schemes", {"age": 24, "category": "maternal_health"})
        self.assertTrue(res_mat.success)
        mat_ids = [s["scheme_id"] for s in res_mat.result.get("matched_schemes", [])]
        self.assertTrue(any("jsy" in sid or "matru" in sid for sid in mat_ids))

    def test_scenario_c_scheme_rules(self):
        """Scenario C: Retrieval of official statutory scheme rules."""
        res = validate_and_execute_tool("get_scheme_rules", {"scheme_id": "ab_pmjay"})
        self.assertTrue(res.success)
        self.assertEqual(res.result.get("status"), "success")
        self.assertIn("5 lakh", res.result.get("benefits", "").lower())
        self.assertTrue(res.result.get("official_url").startswith("https://"))

    def test_scenario_d_document_checklist(self):
        """Scenario D: Retrieval of mandatory verification documents."""
        res = validate_and_execute_tool("get_documents", {"scheme_id": "jsy"})
        self.assertTrue(res.success)
        self.assertEqual(res.result.get("status"), "success")
        docs = res.result.get("documents", [])
        self.assertTrue(any("aadhaar" in d.lower() or "mch card" in d.lower() for d in docs))

    def test_scenario_e_facility_discovery(self):
        """Scenario E: Discovering empanelled healthcare facilities and CSCs."""
        res = validate_and_execute_tool("find_facility", {"district": "Varanasi", "facility_type": "all"})
        self.assertTrue(res.success)
        self.assertEqual(res.result.get("status"), "success")
        self.assertEqual(res.result.get("district"), "Varanasi")
        facilities = res.result.get("facilities", [])
        self.assertGreaterEqual(len(facilities), 2)
        names = [f["name"] for f in facilities]
        self.assertTrue(any("Civil Hospital" in n for n in names))

    def test_scenario_f_grievance_and_escalation(self):
        """Scenario F: Case profiling and human caseworker escalation."""
        # Step 1: Case Summary
        res_summary = validate_and_execute_tool("create_case_summary", {
            "collected_fields": {
                "age": 68,
                "category": "health",
                "primary_need": "Ayushman card rejected at counter",
                "urgency_level": "HIGH"
            }
        })
        self.assertTrue(res_summary.success)
        summary_payload = res_summary.result.get("case_summary", {})
        self.assertEqual(summary_payload.get("user_age"), 68)

        # Step 2: Escalate to human
        res_esc = validate_and_execute_tool("escalate_to_human", {
            "case_summary": summary_payload,
            "reason": "Empanelled hospital refused cashless treatment arbitrarily."
        })
        self.assertTrue(res_esc.success)
        self.assertEqual(res_esc.result.get("status"), "ESCALATED")
        self.assertIn("handoff_package", res_esc.result)

    def test_scenario_g_hallucination_refusal(self):
        """Scenario G: Nonexistent scheme lookup refusal."""
        res = validate_and_execute_tool("get_scheme_rules", {"scheme_id": "fake_free_money_scheme_2026"})
        self.assertTrue(res.success)  # Tool executes without crashing
        self.assertEqual(res.result.get("status"), "not_found")
        self.assertIn("not found", res.result.get("eligibility", "").lower())

    def test_scenario_h_invalid_tool_rejection(self):
        """Scenario H: Strict rejection of unauthorized/hallucinated tool names."""
        for bad_tool in ["get_schemes", "check_eligibility", "arbitrary_custom_tool"]:
            res = validate_and_execute_tool(bad_tool, {"arg": 123})
            self.assertFalse(res.success)
            self.assertIn("not an authorized SanoVocal tool", res.error)

    def test_scenario_i_missing_information_validation(self):
        """Scenario I: Schema validation triggers when required fields are missing."""
        # match_schemes requires 'age' and 'category'
        res = validate_and_execute_tool("match_schemes", {"category": "health"})  # missing age
        self.assertFalse(res.success)
        self.assertIn("validation failed", res.error)

        # get_scheme_rules requires 'scheme_id'
        res_sid = validate_and_execute_tool("get_scheme_rules", {})
        self.assertFalse(res_sid.success)

    def test_scenario_j_voice_pipeline(self):
        """Scenario J: Real Voice Pipeline (Audio -> Whisper STT -> SanoVocal Safety & Tool Routing -> TTS)."""
        # 1. Negative Unit Test: Dummy / invalid audio MUST be rejected (no mock fallback allowed)
        invalid_res = transcribe_audio_bytes(b"RIFFdummybytes", filename="test.wav", language="en")
        self.assertFalse(invalid_res["success"], "Corrupt dummy bytes must be rejected.")
        self.assertIn("insufficient", invalid_res["error"].lower())

        # 2. TTS Unit Test: Real speech synthesis
        speech_bytes = synthesize_speech("Namaste, Ayushman Bharat PM-JAY provides 5 lakh coverage.", language="en")
        self.assertIsInstance(speech_bytes, bytes)
        self.assertGreater(len(speech_bytes), 1000, "TTS audio bytes must be genuine non-empty audio.")

        # 3. Real Audio STT + Safety + Tool Pipeline Round-Trip
        fixture_path = Path("tests/fixtures/test_emergency_query.mp3")
        self.assertTrue(fixture_path.exists(), "Audio test fixture must exist.")
        audio_data = fixture_path.read_bytes()

        # Real Whisper STT
        stt_res = transcribe_audio_bytes(audio_data, filename="emergency.mp3", language="hi")
        self.assertTrue(stt_res["success"], f"Whisper STT failed: {stt_res.get('error')}")
        transcript = stt_res["transcription"].strip()
        self.assertGreater(len(transcript), 0)
        # Ensure it is NOT the old hardcoded fallback sentence
        self.assertNotEqual(transcript, "Hello, I am looking for health insurance schemes for senior citizens.")

        # Real SanoVocal Backend processing of the spoken transcript
        from app.model.inference import generate_assistant_response
        backend_res = generate_assistant_response(transcript)
        # Transcribed respiratory distress and chest pain triggers emergency override
        self.assertEqual(backend_res["safety_verdict"]["category"], "EMERGENCY_OVERRIDE")
        self.assertIn("112", backend_res["response"])

        # Real TTS of the response
        resp_audio = synthesize_speech(backend_res["response"], language="hi")
        self.assertGreater(len(resp_audio), 500)

    def test_storage_sqlite_crud(self):
        """Verifies database initialization and CRUD operations."""
        session_id = "test-sess-12345"
        DataRepository.save_session(session_id, language="en")
        DataRepository.log_message("msg-1", session_id, "user", "Hello test")
        DataRepository.log_message("msg-2", session_id, "assistant", "Hi there")
        DataRepository.log_tool_event(session_id, "match_schemes", {"age": 50}, {"status": "success"}, "SUCCESS")
        case = DataRepository.create_case(
            case_id="CASE-TEST-001",
            session_id=session_id,
            user_name="Ramesh",
            age=52,
            district="Patna",
            category="health"
        )
        self.assertEqual(case["case_id"], "CASE-TEST-001")

        metrics = DataRepository.get_dashboard_metrics()
        self.assertGreaterEqual(metrics["total_sessions"], 1)
        self.assertGreaterEqual(metrics["total_messages"], 2)


if __name__ == "__main__":
    unittest.main()
