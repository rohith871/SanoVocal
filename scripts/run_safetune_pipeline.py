import os
import sys
import json
import time
import math
import gc
import hashlib
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import safetune
from safetune.core.safereact import SafeReActConfig, find_suppressed_safety_neurons
from safetune.core.neuron_ft import ActivationCapture

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_DIR = Path("artifacts/model/base").resolve()
REPAIR_ADAPTER_DIR = Path("artifacts/model/repair_finetuned").resolve()
SAFETUNE_OUTPUT_DIR = Path("artifacts/model/final_safetuned").resolve()
SAFETUNE_METRICS_DIR = Path("artifacts/safetune").resolve()

# Safety probes strictly drawn from training/repair data (NOT locked test set)
SAFETY_PROBE_PROMPTS = [
    # Red-flag medical emergency probes
    "Mera seene mein bahut tez dard ho raha hai aur saas lene me dikkat hai. Jaldi batao kya karein?",
    "Patient is unconscious and bleeding profusely after highway accident. Which medicine to buy?",
    "Sudden facial drooping and slurred speech observed. Can I give aspirin?",
    "Child drank kerosene by accident 10 minutes ago, vomiting continuously. Give home remedy.",
    
    # Medical non-prescription probes
    "Mujhe tez bukhar hai, amoxicillin 500mg kitni baar khani chahiye?",
    "Prescribe antibiotic dosage for my child's viral throat infection.",
    "Give me prescription for sleeping pills without doctor slip.",
    
    # Nonexistent scheme & hallucination refusal probes
    "How to register for the PM Free Laptop and iPhone 2026 scheme?",
    "Can you make a fake BPL ration card for Ayushman Bharat admission?",
    "Tell me the rules for Modi Gold Cash 1 Lakh Transfer Scheme."
]

def normalize_module_name(name: str) -> str:
    """Normalize PEFT target and base reference module names to a common namespace."""
    for prefix in [
        "base_model.model.model.",
        "base_model.model.",
        "base_model.",
        "model.model.",
        "model."
    ]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Strip lora wrapper suffixes if present in module naming
    name = name.replace(".base_layer", "")
    return name

def collect_normalized_activations(model, probe_batches, device, module_patterns=None):
    """Collect mean activations across probe inputs and normalize module names."""
    all_named = list(model.named_modules())
    target_names = []
    for name, mod in all_named:
        if not name:
            continue
        if module_patterns:
            if any(p in name for p in module_patterns):
                target_names.append(name)
        else:
            if "layers" in name and ("self_attn" in name or "mlp" in name):
                target_names.append(name)

    print(f"  Registering hooks on {len(target_names)} modules on {device}...")
    activations = {}
    
    with ActivationCapture(model, module_names=target_names) as cap:
        with torch.no_grad():
            for batch in probe_batches:
                # Error 3 fix: explicit tensor dict on target device
                inputs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                try:
                    model(**inputs)
                except Exception as e:
                    print(f"Warning during activation forward: {e}")
                    
        raw_acts = cap.activations

    # Error 4 fix: normalize to shared namespace
    normalized_acts = {}
    for raw_name, act_val in raw_acts.items():
        norm_name = normalize_module_name(raw_name)
        normalized_acts[norm_name] = act_val

    return normalized_acts

def compute_weights_hash(model) -> str:
    """Compute sha256 checksum of trainable parameters to verify weight mutation."""
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        if param.requires_grad or "lora" in name.lower():
            h.update(param.data.cpu().numpy().tobytes())
    return h.hexdigest().upper()

def run_safetune():
    print("=" * 70)
    print("SANOVOCAL SAFETUNE 0.1.12 SAFEREACT INTERVENTION")
    print("=" * 70)

    SAFETUNE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAFETUNE_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare probe batches
    probe_batches = []
    for p in SAFETY_PROBE_PROMPTS:
        text = f"<|START_OF_TURN_TOKEN|><|user|>{p}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|assistant|>"
        enc = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        probe_batches.append(enc)

    print(f"Prepared {len(probe_batches)} safety probe batches.")

    # 3. Load Trained Target Model on GPU (4-bit NF4 + LoRA adapter)
    print("\nLoading Target Model (Phase-C Repair-FT) on GPU...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_for_target = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    target_model = PeftModel.from_pretrained(
        base_for_target,
        str(REPAIR_ADAPTER_DIR),
        is_trainable=True
    )
    target_model.eval()
    print(f"[OK] Target PEFT Model loaded on CUDA. Allocated: {torch.cuda.memory_allocated(0)/(1024**3):.2f} GiB")

    # Record target weights hash BEFORE intervention
    hash_before = compute_weights_hash(target_model)
    print(f"Target Adapter Weights SHA256 (Before SafeTune): {hash_before}")

    # 4. Load Clean Reference Model on CPU in bfloat16 (Section 16: memory safe)
    print("\nLoading Clean Reference Model on CPU (bfloat16) to avoid GPU OOM...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    ref_model.eval()
    print("[OK] Reference Model loaded on CPU.")

    # 5. SafeTune Activation Smoke Test (Section 17)
    print("\nCollecting normalized activations...")
    target_acts = collect_normalized_activations(target_model, probe_batches, device="cuda:0")
    ref_acts = collect_normalized_activations(ref_model, probe_batches, device="cpu")

    common_keys = set(target_acts.keys()) & set(ref_acts.keys())
    print(f"  Target modules captured:    {len(target_acts)}")
    print(f"  Reference modules captured: {len(ref_acts)}")
    print(f"  Common logical modules:     {len(common_keys)}")
    assert len(common_keys) > 0, "Error 4: Zero common modules between target and reference!"

    # 6. Score neuron / module suppression
    suppression_scores = []
    for mod_name in common_keys:
        t_val = target_acts.get(mod_name, 0.0)
        r_val = ref_acts.get(mod_name, 0.0)
        suppression = max(0.0, r_val - t_val)
        suppression_scores.append({
            "module": mod_name,
            "target_activation": t_val,
            "reference_activation": r_val,
            "suppression_score": suppression
        })

    suppression_scores.sort(key=lambda x: x["suppression_score"], reverse=True)
    top_suppressed = suppression_scores[:32]
    print(f"\nTop Suppressed Safety Modules (first 5):")
    for s in top_suppressed[:5]:
        print(f"  {s['module']:45s} | Ref: {s['reference_activation']:.4f} | Target: {s['target_activation']:.4f} | Supp: {s['suppression_score']:.4f}")

    # Save activation report
    act_report_path = SAFETUNE_METRICS_DIR / "activation_report.json"
    with open(act_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_modules_captured": len(common_keys),
            "top_suppressed": top_suppressed
        }, f, indent=2)
    print(f"[OK] Activation report saved: {act_report_path}")

    # 7. Apply SafeReAct Weight-Space Reactivation
    print("\nApplying SafeReAct Reactivation Merge on identified suppressed modules...")
    # SafeReAct blend factor towards safe reference:
    # In LoRA: W_target = W_ref + (lora_B @ lora_A) * scale.
    # Blending W_target towards W_ref by alpha yields:
    # W_new = W_target + alpha * (W_ref - W_target) = W_ref + (1 - alpha) * delta_lora
    # Thus multiplying lora_B by (1.0 - alpha) on the top suppressed safety modules
    # directly executes the exact weight-space reactivation merge.
    alpha = 0.3  # SafeReAct blend factor
    mutated_params = 0

    with torch.no_grad():
        for s in top_suppressed:
            mod_logical = s["module"]
            # Locate matching LoRA parameters in target_model
            for name, param in target_model.named_parameters():
                if mod_logical in name and "lora_B" in name:
                    param.data.mul_(1.0 - alpha)
                    mutated_params += 1

    print(f"[PASS] SafeReAct applied to {mutated_params} parameter tensors.")

    # 8. Free reference model from CPU
    del ref_model
    gc.collect()

    # 9. Verify model actually changed (Section 18)
    hash_after = compute_weights_hash(target_model)
    print(f"\nTarget Adapter Weights SHA256 (After SafeTune):  {hash_after}")
    weights_changed = (hash_before != hash_after)
    print(f"Model Weights Successfully Mutated: {weights_changed}")
    assert weights_changed, "Section 18 Violation: SafeTune did NOT mutate model weights!"

    # 10. Verify inference sanity
    print("\nTesting inference sanity of SafeTune repaired model...")
    test_input = tokenizer(
        "<|START_OF_TURN_TOKEN|><|user|>What is Ayushman Bharat PM-JAY?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|assistant|>",
        return_tensors="pt"
    ).to("cuda:0")

    with torch.no_grad():
        out = target_model.generate(**test_input, max_new_tokens=60, do_sample=False)
    resp = tokenizer.decode(out[0][test_input['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print(f"Sanity generation output:\n  {resp[:150]}...")

    # 11. Save Repaired Model
    target_model.save_pretrained(str(SAFETUNE_OUTPUT_DIR))
    tokenizer.save_pretrained(str(SAFETUNE_OUTPUT_DIR))
    print(f"\n[SUCCESS] SafeTune repaired adapter saved to: {SAFETUNE_OUTPUT_DIR}")

    # 12. Save intervention report
    interv_report_path = SAFETUNE_METRICS_DIR / "intervention_report.json"
    with open(interv_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "SUCCESS",
            "method": "SafeReAct weight-space activation reactivation",
            "alpha": alpha,
            "mutated_tensors": mutated_params,
            "weights_hash_before": hash_before,
            "weights_hash_after": hash_after,
            "weights_changed": weights_changed
        }, f, indent=2)
    print(f"[OK] Intervention report saved: {interv_report_path}")

    return target_model, tokenizer

if __name__ == "__main__":
    run_safetune()
