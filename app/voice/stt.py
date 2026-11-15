"""
SanoVocal Real Speech-to-Text (STT) Module
Uses faster-whisper with Singleton model caching, multilingual support (English/Hindi/Hinglish),
automatic language detection, and safe fallback from CUDA to CPU int8.
"""

import os
import io
import time
import uuid
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Safe patch for PyAV metadata_errors compatibility across av versions
try:
    import av
    import faster_whisper.audio as _fw_audio
    _orig_av_open = av.open
    _fw_audio.av.open = lambda f, mode='r', **kwargs: _orig_av_open(f, mode=mode)
except Exception:
    pass

_WHISPER_LOCK = threading.Lock()
_WHISPER_MODEL = None
_WHISPER_DEVICE = None
_DEFAULT_MODEL_NAME = os.getenv("SANOVOCAL_WHISPER_MODEL", "base")

def get_whisper_model(model_name: Optional[str] = None, force_cpu: bool = False) -> Tuple[Any, str]:
    """
    Singleton loader for faster-whisper.
    Loads once, caches in memory, and reuses across all requests.
    Attempts CUDA float16 first; gracefully falls back to CPU int8.
    """
    global _WHISPER_MODEL, _WHISPER_DEVICE
    target_model = model_name or _DEFAULT_MODEL_NAME

    with _WHISPER_LOCK:
        if _WHISPER_MODEL is not None and not force_cpu:
            return _WHISPER_MODEL, _WHISPER_DEVICE

        from faster_whisper import WhisperModel

        device_pref = os.getenv("SANOVOCAL_WHISPER_DEVICE", "cpu").lower()

        # Try CUDA if preferred and not forced to CPU
        if not force_cpu and device_pref in ["cuda"]:
            try:
                print(f"[SanoVocal STT] Initializing faster-whisper ({target_model}) on CUDA (float16)...")
                model = WhisperModel(target_model, device="cuda", compute_type="float16")
                _WHISPER_MODEL = model
                _WHISPER_DEVICE = "cuda"
                print(f"[SanoVocal STT] Loaded {target_model} on CUDA.")
                return _WHISPER_MODEL, _WHISPER_DEVICE
            except Exception as cuda_err:
                print(f"[SanoVocal STT] CUDA Whisper initialization failed ({cuda_err}). Falling back to CPU...")

        # CPU mode (int8 quantization for fast execution and 0 VRAM overhead)
        print(f"[SanoVocal STT] Initializing faster-whisper ({target_model}) on CPU (int8)...")
        model = WhisperModel(target_model, device="cpu", compute_type="int8")
        _WHISPER_MODEL = model
        _WHISPER_DEVICE = "cpu"
        print(f"[SanoVocal STT] Loaded {target_model} on CPU.")
        return _WHISPER_MODEL, _WHISPER_DEVICE

def transcribe_audio_bytes(
    audio_bytes: bytes,
    filename: str = "input.wav",
    language: str = "auto"
) -> Dict[str, Any]:
    """
    Transcribes audio bytes using faster-whisper.
    Returns a structured dictionary with transcription, language metadata, and latency.
    Raises or returns structured error if audio is invalid or empty.
    """
    start_time = time.time()

    # Reject empty or corrupt dummy headers
    if not audio_bytes or len(audio_bytes) < 64:
        return {
            "success": False,
            "transcription": "",
            "language": None,
            "latency_ms": 0,
            "error": "Empty or insufficient audio data provided."
        }

    # Determine file extension
    ext = Path(filename).suffix.lower()
    if not ext or ext not in [".wav", ".mp3", ".ogg", ".m4a", ".webm"]:
        ext = ".wav"

    temp_id = uuid.uuid4().hex[:8]
    temp_file = Path(f"temp_stt_{temp_id}{ext}").resolve()

    try:
        temp_file.write_bytes(audio_bytes)

        # Normalize language parameter
        lang_lower = (language or "auto").lower().strip()
        if lang_lower in ["hi", "hindi"]:
            lang_param = "hi"
        elif lang_lower in ["en", "english"]:
            lang_param = "en"
        else:
            lang_param = None  # auto-detection (handles Hindi, English, and Hinglish code-mixing)

        model, dev = get_whisper_model()
        try:
            segments, info = model.transcribe(
                str(temp_file),
                language=lang_param,
                task="transcribe",
                beam_size=3
            )
            transcript_parts = [seg.text for seg in segments]
        except Exception as first_err:
            if dev == "cuda":
                print(f"[SanoVocal STT] CUDA transcription error ({first_err}). Retrying on CPU...")
                model, dev = get_whisper_model(force_cpu=True)
                segments, info = model.transcribe(
                    str(temp_file),
                    language=lang_param,
                    task="transcribe",
                    beam_size=3
                )
                transcript_parts = [seg.text for seg in segments]
            else:
                raise first_err

        final_text = " ".join(transcript_parts).strip()
        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "transcription": final_text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
            "device": dev,
            "latency_ms": latency_ms,
            "error": None
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "transcription": "",
            "language": None,
            "latency_ms": latency_ms,
            "error": f"Transcription error: {str(e)}"
        }

    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass

def is_stt_available() -> bool:
    try:
        model, _ = get_whisper_model()
        return model is not None
    except Exception:
        return False
