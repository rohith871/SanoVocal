"""
SanoVocal End-to-End Multilingual Smoke Test & API Verification
"""

import unittest
from fastapi.testclient import TestClient
from app.api.main import app
from app.core.safety import evaluate_deterministic_safety
from app.tools.registry import validate_and_execute_tool

class TestEndToEndSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "healthy")

    def test_system_info_endpoint(self):
        resp = self.client.get("/v1/system/info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("CohereLabs/tiny-aya-global", data["model_architecture"])
        self.assertIn("emergency_screen", data["authorized_tools"])

    def test_multilingual_emergency_hindi(self):
        # Hindi emergency triage
        query_hi = "रोगी बेहोश है और छाती में तेज दर्द हो रहा है"
        # Deterministic safety
        verdict = evaluate_deterministic_safety("behosh aur chhati me tez dard")
        self.assertFalse(verdict.is_safe)
        self.assertEqual(verdict.category, "EMERGENCY_OVERRIDE")

    def test_hinglish_scheme_matching(self):
        # Hinglish profile match
        res = validate_and_execute_tool("match_schemes", {
            "age": 71,
            "category": "health",
            "income_status": "eligible"
        })
        self.assertTrue(res.success)
        matched_ids = [s["scheme_id"] for s in res.result["matched_schemes"]]
        self.assertIn("ayushman_vay_vandana", matched_ids)

    def test_api_case_creation_and_listing(self):
        create_payload = {
            "session_id": "test-e2e-session",
            "user_name": "Anita Verma",
            "age": 28,
            "district": "Lucknow",
            "category": "maternal_health",
            "matched_schemes": ["jsy", "jssk"],
            "escalation_reason": "CSC operator charged unauthorized fees.",
            "urgency_level": "HIGH"
        }
        res_post = self.client.post("/v1/chat/cases", json=create_payload)
        self.assertEqual(res_post.status_code, 200)
        data = res_post.json()
        self.assertTrue(data["case_id"].startswith("CASE-"))

        res_get = self.client.get("/v1/chat/cases")
        self.assertEqual(res_get.status_code, 200)
        cases = res_get.json()
        self.assertTrue(any(c["session_id"] == "test-e2e-session" for c in cases))

if __name__ == "__main__":
    unittest.main()
