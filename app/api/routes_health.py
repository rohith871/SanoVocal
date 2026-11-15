"""
SanoVocal Health & System Inspection Routes
"""

from fastapi import APIRouter
from app.core.config import BASE_MODEL_NAME, V3_ADAPTER_PATH, V3_ADAPTER_SHA256, LOCKED_TEST_SHA256, OFFICIAL_TOOL_NAMES
from app.model.loader import get_device_info, is_model_loaded

router = APIRouter(prefix="", tags=["System"])

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SanoVocal Production Engine",
        "version": "3.0.0"
    }

@router.get("/v1/system/info")
def system_info():
    device_info = get_device_info()
    return {
        "model_architecture": "CohereLabs/tiny-aya-global + SanoVocal V3 LoRA Adapter",
        "frozen_v3_adapter_sha256": V3_ADAPTER_SHA256,
        "locked_test_sha256": LOCKED_TEST_SHA256,
        "authorized_tools": sorted(list(OFFICIAL_TOOL_NAMES)),
        "hardware": device_info,
        "adapter_path_exists": V3_ADAPTER_PATH.exists()
    }
