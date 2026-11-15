"""
SanoVocal Model Manager (Singleton)
Loads the frozen Base Tiny Aya Global + V3 LoRA adapter once into memory.
Guarantees 4-bit NF4 QLoRA inference on RTX 3050 or safe CPU fallback.
"""

import sys
import threading
from typing import Tuple, Optional, Any
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from app.core.config import BASE_MODEL_NAME, V3_ADAPTER_PATH

_MODEL_LOCK = threading.Lock()
_TOKENIZER = None
_MODEL = None
_DEVICE_INFO = {}

def get_device_info() -> dict:
    global _DEVICE_INFO
    if not _DEVICE_INFO:
        has_cuda = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if has_cuda else 0.0
        _DEVICE_INFO = {
            "cuda_available": has_cuda,
            "device_name": device_name,
            "vram_gb": round(vram, 2),
            "model_loaded": _MODEL is not None
        }
    else:
        _DEVICE_INFO["model_loaded"] = _MODEL is not None
    return _DEVICE_INFO

def load_v3_model(device_preference: str = "auto") -> Tuple[Any, Any]:
    """
    Singleton loader for SanoVocal V3 (Base Tiny Aya + V3 LoRA adapter).
    Thread-safe and cached.
    """
    global _TOKENIZER, _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _TOKENIZER, _MODEL

        print("[SanoVocal ModelLoader] Loading Tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME,
                trust_remote_code=True,
                use_fast=False,
                local_files_only=True
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME,
                trust_remote_code=True,
                use_fast=False
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        has_cuda = torch.cuda.is_available() and device_preference != "cpu"

        if has_cuda:
            print("[SanoVocal ModelLoader] Initializing 4-bit NF4 Base Model on CUDA...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    local_files_only=True
                )
            except Exception:
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
        else:
            print("[SanoVocal ModelLoader] Initializing Base Model on CPU...")
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    local_files_only=True
                )
            except Exception:
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )

        adapter_path = str(V3_ADAPTER_PATH)
        if Path(adapter_path).exists():
            print(f"[SanoVocal ModelLoader] Attaching frozen V3 LoRA adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            print(f"[SanoVocal ModelLoader] Warning: Adapter path {adapter_path} not found. Running base model.")
            model = base_model

        model.eval()
        _TOKENIZER = tokenizer
        _MODEL = model
        print("[SanoVocal ModelLoader] Model and Tokenizer successfully loaded and cached.")
        return _TOKENIZER, _MODEL

def is_model_loaded() -> bool:
    return _MODEL is not None
