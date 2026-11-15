"""
SanoVocal Voice API Routes
Provides endpoints for audio transcription, voice synthesis,
and unified end-to-end voice processing (Audio -> STT -> SanoVocal -> TTS).
"""

import uuid
from fastapi import APIRouter, UploadFile, File, Form, Response, HTTPException
from app.voice.stt import transcribe_audio_bytes
from app.voice.tts import synthesize_speech_detailed
from app.model.inference import generate_assistant_response
from app.core.session import SessionContext
from app.storage.repository import DataRepository

router = APIRouter(prefix="/v1/voice", tags=["Voice"])

@router.post("/transcribe")
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")

    result = transcribe_audio_bytes(audio_bytes, filename=file.filename or "input.wav", language=language)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Transcription failed."))

    return {
        "filename": file.filename,
        "language": result.get("language"),
        "language_probability": result.get("language_probability"),
        "transcription": result.get("transcription"),
        "latency_ms": result.get("latency_ms"),
        "device": result.get("device")
    }

@router.post("/synthesize")
def synthesize_speech_endpoint(
    text: str = Form(...),
    language: str = Form("en")
):
    tts_result = synthesize_speech_detailed(text, language=language)
    if not tts_result.get("success"):
        raise HTTPException(status_code=500, detail=tts_result.get("error", "Speech synthesis failed."))

    return Response(
        content=tts_result["audio_bytes"],
        media_type="audio/mpeg",
        headers={
            "X-TTS-Latency-MS": str(tts_result.get("latency_ms", 0)),
            "X-TTS-Language": str(tts_result.get("language", "en"))
        }
    )

@router.post("/process")
async def process_voice_turn_endpoint(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    session_id: str = Form(None)
):
    """
    End-to-End Voice Turn:
    1. Audio -> Whisper STT (transcription + language detection)
    2. Transcription -> SanoVocal Chat Pipeline (Safety -> Model -> Strict Tool Execution -> Grounded Response)
    3. Response Text -> gTTS Synthesis
    Returns full structured response including transcription, chat response, tool events, and synthesized audio.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")

    # Step 1: STT
    stt_res = transcribe_audio_bytes(audio_bytes, filename=file.filename or "input.wav", language=language)
    if not stt_res.get("success"):
        raise HTTPException(status_code=400, detail=f"Voice STT failed: {stt_res.get('error')}")

    user_query = stt_res.get("transcription", "").strip()
    detected_lang = stt_res.get("language", "en")

    # Step 2: SanoVocal Core Pipeline
    sess_id = session_id or str(uuid.uuid4())
    session = SessionContext(session_id=sess_id, language=detected_lang)

    chat_res = generate_assistant_response(user_text=user_query, session=session)

    # Persist in DB
    DataRepository.save_session(sess_id, language=detected_lang)
    DataRepository.log_message(str(uuid.uuid4()), sess_id, "user", user_query)
    DataRepository.log_message(
        str(uuid.uuid4()), sess_id, "assistant", chat_res["response"],
        tool_calls=[chat_res["tool_call"]] if chat_res["tool_call"] else None,
        tool_results=[chat_res["tool_result"]] if chat_res["tool_result"] else None
    )

    # Step 3: TTS
    tts_res = synthesize_speech_detailed(chat_res["response"], language=detected_lang)

    total_latency = stt_res.get("latency_ms", 0) + chat_res.get("latency_ms", 0) + tts_res.get("latency_ms", 0)

    return {
        "session_id": sess_id,
        "transcription": user_query,
        "detected_language": detected_lang,
        "response_text": chat_res["response"],
        "safety_verdict": chat_res.get("safety_verdict"),
        "tool_call": chat_res.get("tool_call"),
        "tool_result": chat_res.get("tool_result"),
        "audio_generated": tts_res.get("success", False),
        "latencies": {
            "stt_ms": stt_res.get("latency_ms", 0),
            "inference_ms": chat_res.get("latency_ms", 0),
            "tts_ms": tts_res.get("latency_ms", 0),
            "total_roundtrip_ms": total_latency
        }
    }
