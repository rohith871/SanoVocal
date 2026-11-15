"""
SanoVocal Dedicated Voice Suite: Unit, Integration, and E2E Tests
Tests Whisper STT, gTTS Speech Synthesis, Language Handling, and FastAPI Voice Endpoints.
"""

import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from app.voice.stt import transcribe_audio_bytes, is_stt_available, get_whisper_model
from app.voice.tts import synthesize_speech, synthesize_speech_detailed, is_tts_available
from app.model.inference import generate_assistant_response
from app.api.main import app


class TestVoicePipelineFull(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.scheme_fixture = Path("tests/fixtures/test_scheme_query.mp3")
        cls.emergency_fixture = Path("tests/fixtures/test_emergency_query.mp3")

    # -------------------------------------------------------------------------
    # 1. UNIT TESTS: STT
    # -------------------------------------------------------------------------
    def test_stt_empty_and_insufficient_audio_rejected(self):
        """Verifies that empty and corrupt audio bytes return explicit structured errors."""
        res_empty = transcribe_audio_bytes(b"")
        self.assertFalse(res_empty["success"])
        self.assertIn("insufficient", res_empty["error"].lower())

        res_corrupt = transcribe_audio_bytes(b"RIFFdummybytes", filename="corrupt.wav")
        self.assertFalse(res_corrupt["success"])
        self.assertIn("insufficient", res_corrupt["error"].lower())
        # Crucial: Must NEVER return a hardcoded fallback string
        self.assertEqual(res_corrupt["transcription"], "")

    def test_stt_valid_audio_transcription(self):
        """Verifies that real audio fixture produces a valid transcription with faster-whisper."""
        self.assertTrue(self.scheme_fixture.exists())
        audio_data = self.scheme_fixture.read_bytes()
        res = transcribe_audio_bytes(audio_data, filename="scheme.mp3", language="hi")
        self.assertTrue(res["success"], f"Transcription failed: {res.get('error')}")
        self.assertGreater(len(res["transcription"]), 10)
        self.assertNotEqual(res["transcription"], "Hello, I am looking for health insurance schemes for senior citizens.")
        self.assertIn("language", res)
        self.assertGreater(res["latency_ms"], 0)

    def test_stt_language_auto_detection(self):
        """Verifies automatic language detection when language='auto'."""
        audio_data = self.scheme_fixture.read_bytes()
        res = transcribe_audio_bytes(audio_data, filename="scheme.mp3", language="auto")
        self.assertTrue(res["success"])
        self.assertIsNotNone(res["language"])
        self.assertGreater(res["language_probability"], 0.4)

    # -------------------------------------------------------------------------
    # 2. UNIT TESTS: TTS
    # -------------------------------------------------------------------------
    def test_tts_playable_audio_generation(self):
        """Verifies that gTTS generates valid playable MP3 frames."""
        res = synthesize_speech_detailed("Namaste, Ayushman Bharat PM-JAY scheme provides coverage.", language="en")
        self.assertTrue(res["success"])
        self.assertGreater(res["byte_count"], 1000)
        self.assertEqual(res["audio_format"], "mp3")
        # Check standard MP3 frame sync bytes
        self.assertEqual(res["audio_bytes"][:2], b'\xff\xf3')

    def test_tts_empty_text_handled(self):
        """Verifies empty text handling."""
        res = synthesize_speech_detailed("   ", language="en")
        self.assertFalse(res["success"])
        self.assertEqual(res["audio_bytes"], b"")
        self.assertIn("empty", res["error"].lower())

    # -------------------------------------------------------------------------
    # 3. INTEGRATION TESTS: Audio -> STT -> SanoVocal Core
    # -------------------------------------------------------------------------
    def test_voice_integration_emergency_triage(self):
        """
        Integration:
        Audio (Emergency) -> Whisper STT -> SanoVocal Safety Middleware -> Emergency Override
        """
        self.assertTrue(self.emergency_fixture.exists())
        audio_data = self.emergency_fixture.read_bytes()

        stt_res = transcribe_audio_bytes(audio_data, filename="emergency.mp3", language="hi")
        self.assertTrue(stt_res["success"])
        transcript = stt_res["transcription"]

        # Run through SanoVocal core pipeline
        chat_res = generate_assistant_response(transcript)
        # Priority #1: Clinical Emergency Red-Flag Triggered
        self.assertEqual(chat_res["safety_verdict"]["category"], "EMERGENCY_OVERRIDE")
        self.assertIn("112", chat_res["response"])

        # Generate Audio Response
        tts_res = synthesize_speech_detailed(chat_res["response"], language="hi")
        self.assertTrue(tts_res["success"])
        self.assertGreater(len(tts_res["audio_bytes"]), 500)

    # -------------------------------------------------------------------------
    # 4. API TESTS: FastAPI Voice Endpoints
    # -------------------------------------------------------------------------
    def test_api_voice_transcribe_endpoint(self):
        audio_data = self.scheme_fixture.read_bytes()
        files = {"file": ("scheme.mp3", audio_data, "audio/mpeg")}
        resp = self.client.post("/v1/voice/transcribe", files=files, data={"language": "hi"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("transcription", data)
        self.assertGreater(len(data["transcription"]), 5)

    def test_api_voice_synthesize_endpoint(self):
        resp = self.client.post("/v1/voice/synthesize", data={"text": "Aadhaar card required.", "language": "en"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "audio/mpeg")
        self.assertGreater(len(resp.content), 500)

    def test_api_voice_process_e2e_endpoint(self):
        audio_data = self.emergency_fixture.read_bytes()
        files = {"file": ("emergency.mp3", audio_data, "audio/mpeg")}
        resp = self.client.post("/v1/voice/process", files=files, data={"language": "hi"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("transcription", data)
        self.assertIn("response_text", data)
        self.assertTrue(data["audio_generated"])
        self.assertIn("latencies", data)
        self.assertGreater(data["latencies"]["total_roundtrip_ms"], 0)


if __name__ == "__main__":
    unittest.main()
