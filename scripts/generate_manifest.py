import os
import sys
import json
import time
import hashlib
from pathlib import Path
import torch
import importlib.metadata as md

sys.stdout.reconfigure(encoding='utf-8')

MANIFEST_DIR = Path("artifacts/manifests").resolve()
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = MANIFEST_DIR / "final_pipeline_manifest.json"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

def create_manifest():
    print("=" * 70)
    print("GENERATING SANOVOCAL FINAL PIPELINE MANIFEST")
    print("=" * 70)

    # 1. Datasets
    repair_path = Path("data/repair/sanovocal_tool_contract_repair.jsonl").resolve()
    test_path = Path("data/traces/agent_trajectories_test.jsonl").resolve()

    repair_hash = sha256_file(repair_path)
    test_hash = sha256_file(test_path)

    # 2. Package versions
    pkg_versions = {}
    for pkg in ["transformers", "peft", "trl", "bitsandbytes", "agenttune", "safetune", "auditkit"]:
        try:
            pkg_versions[pkg] = md.version(pkg)
        except Exception:
            pkg_versions[pkg] = "unknown"

    # 3. Metrics if available
    baseline_metrics = None
    b_file = Path("artifacts/metrics/baseline_45.json")
    if b_file.exists():
        with open(b_file, "r", encoding="utf-8") as f:
            baseline_metrics = json.load(f)["metrics"]

    ft_metrics = None
    ft_file = Path("artifacts/metrics/repair_ft_45.json")
    if ft_file.exists():
        with open(ft_file, "r", encoding="utf-8") as f:
            ft_metrics = json.load(f)["metrics"]

    st_metrics = None
    st_file = Path("artifacts/metrics/safetune_final_45.json")
    if st_file.exists():
        with open(st_file, "r", encoding="utf-8") as f:
            st_metrics = json.load(f)["metrics"]

    auditkit_metrics = None
    ak_file = Path("artifacts/auditkit/final_audit_report.json")
    if ak_file.exists():
        with open(ak_file, "r", encoding="utf-8") as f:
            auditkit_metrics = json.load(f)

    # 4. Final Adapter Hash
    final_adapter_path = Path("artifacts/model/final_safetuned/adapter_model.safetensors")
    final_adapter_hash = sha256_file(final_adapter_path) if final_adapter_path.exists() else None

    manifest = {
        "manifest_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": "CohereLabs/tiny-aya-global",
        "local_model_path": str(Path("artifacts/model/base").resolve()),
        "tokenizer_version": pkg_versions.get("transformers"),
        "transformers_version": pkg_versions.get("transformers"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "n/a",
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a",
        "agenttune_version": pkg_versions.get("agenttune"),
        "safetune_version": pkg_versions.get("safetune"),
        "auditkit_version": pkg_versions.get("auditkit"),
        "peft_version": pkg_versions.get("peft"),
        "trl_version": pkg_versions.get("trl"),

        "repair_dataset": {
            "path": str(repair_path),
            "sha256": repair_hash,
            "record_count": 226
        },
        "locked_test_dataset": {
            "path": str(test_path),
            "sha256": test_hash,
            "record_count": 45,
            "hold_out_guarantee": "Never entered training, validation, or SafeTune probe sets."
        },
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "torch.bfloat16",
            "bnb_4bit_use_double_quant": True
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "task_type": "CAUSAL_LM"
        },
        "training_config": {
            "framework": "agenttune.create_distill_trainer / Project.distill()",
            "epochs": 3,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "optimizer": "adamw_torch"
        },
        "evaluation_metrics": {
            "baseline": baseline_metrics,
            "repair_finetuned": ft_metrics,
            "safetune_repaired": st_metrics,
            "auditkit_report": auditkit_metrics
        },
        "final_model": {
            "path": str(Path("artifacts/model/final_safetuned").resolve()),
            "adapter_weights_sha256": final_adapter_hash
        }
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Manifest successfully written to: {OUT_FILE}")
    return manifest

if __name__ == "__main__":
    create_manifest()
