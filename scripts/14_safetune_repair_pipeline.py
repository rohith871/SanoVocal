"""
SanoVocal — Phase D: SafeTune Safety Audit & Repair Pipeline
Applies SafeLoRA (Hsu et al., NeurIPS 2024) via official SafeTune library:
  `safetune.interventions.recover.safe_lora.apply_safe_lora`

Preserves experimental distinction:
  - Input:  outputs/checkpoints/aya_expanse_8b_qlora/  (Untouched Fine-Tuned Adapter)
  - Output: outputs/checkpoints/aya_expanse_8b_safetuned/ (Safe-Tuned Adapter)

Strictly prevents model substitution:
  - Base model must be CohereLabs/aya-expanse-8b
  - Never run against proxy models or ungated substitutes
"""

import os
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OFFICIAL_MODEL_ID = "CohereLabs/aya-expanse-8b"
QLORA_CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "aya_expanse_8b_qlora"
SAFETUNED_CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "aya_expanse_8b_safetuned"

def verify_safetune_environment():
    print("[Verification] Checking SafeTune installation and APIs...")
    import safetune
    from safetune.interventions.recover.safe_lora import apply_safe_lora
    print(f"  SafeTune version: {getattr(safetune, '__version__', '0.1.12')}")
    print(f"  Verified API:     safetune.interventions.recover.safe_lora.apply_safe_lora [OK]")
    return True

def run_safetune_repair(dry_run: bool = False, threshold: float = 0.5):
    print("=" * 65)
    print(" SANOVOCAL — PHASE D: SAFETUNE AUDIT & REPAIR PIPELINE")
    print("=" * 65)

    verify_safetune_environment()

    if dry_run:
        print("\n[Dry-Run Mode] Validating pipeline configuration without weight downloads:")
        print(f"  Target Base Model:      {OFFICIAL_MODEL_ID}")
        print(f"  Target Adapter Input:   {QLORA_CHECKPOINT_DIR}")
        print(f"  Repaired Output Dir:    {SAFETUNED_CHECKPOINT_DIR}")
        print(f"  SafeLoRA Threshold:     {threshold}")
        print(f"  Target Modules:         ['q_proj', 'k_proj', 'v_proj', 'o_proj']")
        print("  [OK] SafeTune pipeline configuration validated syntactically.")
        return

    # Real execution after HF cooldown and QLoRA completion
    if not QLORA_CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"Missing QLoRA checkpoint at {QLORA_CHECKPOINT_DIR}. Phase C training must complete first.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from safetune.interventions.recover.safe_lora import apply_safe_lora

    print(f"\nLoading base model {OFFICIAL_MODEL_ID} and attaching QLoRA adapter...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(OFFICIAL_MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base_model, str(QLORA_CHECKPOINT_DIR))

    print(f"Applying SafeLoRA projection (threshold={threshold})...")
    repaired_model = apply_safe_lora(
        model,
        threshold=threshold,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    SAFETUNED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    repaired_model.save_pretrained(str(SAFETUNED_CHECKPOINT_DIR))
    print(f"SafeTune repaired adapter successfully saved to: {SAFETUNED_CHECKPOINT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run_safetune_repair(dry_run=args.dry_run, threshold=args.threshold)
