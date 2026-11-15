"""
SanoVocal Model Manager (Singleton)
Loads the frozen Base Tiny Aya Global + LoRA adapter once into memory.
Guarantees 4-bit NF4 QLoRA inference on CUDA or safe CPU fallback.
Loads from local weights if present, otherwise streams from Hugging Face Hub.
"""

import os
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
    Singleton loader for SanoVocal (Base Tiny Aya + LoRA adapter).
    Thread-safe and cached.
    """
    global _TOKENIZER, _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _TOKENIZER, _MODEL

        # Read optional HF token from environment or Streamlit secrets
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
        if not hf_token:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
                    hf_token = st.secrets["HF_TOKEN"]
            except Exception:
                pass

        # Check if local base model weights actually exist on disk
        local_base = Path("artifacts/model/base")
        has_local_base_weights = local_base.exists() and any(local_base.glob("*.safetensors"))
        base_model_id = str(local_base) if has_local_base_weights else "CohereLabs/tiny-aya-global"

        # Check if local adapter weights actually exist on disk
        local_adapter = Path(str(V3_ADAPTER_PATH))
        has_local_adapter_weights = (
            local_adapter.exists()
            and ((local_adapter / "adapter_model.safetensors").exists() or (local_adapter / "adapter_model.bin").exists())
        )
        adapter_id = str(local_adapter) if has_local_adapter_weights else "rohith871/sanovocal-adapter"

        print(f"[SanoVocal ModelLoader] Loading Tokenizer from {base_model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                use_fast=False,
                token=hf_token,
                local_files_only=has_local_base_weights
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                use_fast=False,
                token=hf_token
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        has_cuda = torch.cuda.is_available() and device_preference != "cpu"

        if has_cuda:
            print(f"[SanoVocal ModelLoader] Initializing 4-bit NF4 Base Model on CUDA from {base_model_id}...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    token=hf_token,
                    local_files_only=has_local_base_weights
                )
            except Exception:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    token=hf_token
                )
        else:
            print(f"[SanoVocal ModelLoader] Initializing Base Model on CPU from {base_model_id}...")
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    token=hf_token,
                    local_files_only=has_local_base_weights
                )
            except Exception:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    token=hf_token
                )

        print(f"[SanoVocal ModelLoader] Attaching LoRA adapter from {adapter_id}...")
        try:
            model = PeftModel.from_pretrained(
                base_model,
                adapter_id,
                token=hf_token
            )
        except Exception as e:
            print(f"[SanoVocal ModelLoader] Warning: Failed to load adapter from {adapter_id} ({e}). Running base model.")
            model = base_model

        model.eval()
        _TOKENIZER = tokenizer
        _MODEL = model
        print("[SanoVocal ModelLoader] Model and Tokenizer successfully loaded and cached.")
        return _TOKENIZER, _MODEL

def is_model_loaded() -> bool:
    return _MODEL is not None
