"""
SanoVocal Text-to-Speech (TTS) Module
Provides modular speech synthesis in English and Hindi.
Currently powered by gTTS (Google Text-to-Speech).
NOTE: Requires active internet connectivity for cloud synthesis.
"""

import io
import time
from typing import Dict, Any, Optional
from gtts import gTTS

def synthesize_speech(text: str, language: str = "en") -> bytes:
    """
    Synthesizes speech audio from text using gTTS.
    Returns raw MP3 audio bytes.
    Supported language codes: 'en' (English), 'hi' (Hindi/Hinglish).
    """
    res = synthesize_speech_detailed(text, language=language)
    return res.get("audio_bytes", b"")

def synthesize_speech_detailed(text: str, language: str = "en") -> Dict[str, Any]:
    """
    Synthesizes speech audio and returns structured metadata including latency and error details.
    """
    start_time = time.time()
    clean_text = text.replace("*", "").replace("#", "").strip()

    if not clean_text:
        return {
            "success": False,
            "audio_bytes": b"",
            "language": language,
            "latency_ms": 0,
            "error": "Empty text provided for synthesis."
        }

    # Map language
    lang_lower = (language or "en").lower().strip()
    lang_code = "hi" if lang_lower in ["hi", "hindi", "hinglish"] else "en"

    fp = io.BytesIO()
    try:
        # Cap length to avoid excessive cloud payload
        tts = gTTS(text=clean_text[:500], lang=lang_code, slow=False)
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_data = fp.read()
        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "audio_bytes": audio_data,
            "language": lang_code,
            "audio_format": "mp3",
            "byte_count": len(audio_data),
            "latency_ms": latency_ms,
            "error": None
        }
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "audio_bytes": b"",
            "language": lang_code,
            "latency_ms": latency_ms,
            "error": f"TTS synthesis failed (Check internet connection): {str(e)}"
        }

def is_tts_available() -> bool:
    try:
        from gtts import gTTS
        return True
    except ImportError:
        return False
