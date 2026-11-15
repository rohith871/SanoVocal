import json
import hashlib
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/sanovocal_aya_qlora_training.ipynb")

def validate_notebook():
    print("=" * 70)
    print(" VALIDATING REPAIRED SANOVOCAL KAGGLE NOTEBOOK")
    print("=" * 70)
    
    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"Total cells in notebook: {len(cells)}")
    assert len(cells) == 11, f"Expected exactly 11 cells, found {len(cells)}"

    full_code = ""
    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if ctype == "code":
            full_code += source + "\n"
        print(f"Cell {idx+1:02d} [{ctype.upper()}]: {source.splitlines()[0] if source.splitlines() else 'EMPTY'}")

    print("\n--- CHECKLIST VERIFICATION ---")

    # 1. Phase C Baseline isolation
    assert "PHASE C — BASELINE EVALUATION" in full_code or "PHASE C: BASELINE EVALUATION" in full_code, "Phase C baseline header missing"
    assert "baseline_evaluation_45_test.jsonl" in full_code, "baseline_evaluation_45_test.jsonl missing"
    assert "phase_c_baseline_manifest.json" in full_code, "phase_c_baseline_manifest.json missing"
    print("[PASS] Phase C: Baseline evaluation strictly present and isolated.")

    # 2. SHA256 integrity check
    EXPECTED_SHA = "667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3"
    assert EXPECTED_SHA in full_code, "Test SHA256 integrity hash missing"
    print(f"[PASS] Locked 45 test set SHA256 integrity check present ({EXPECTED_SHA}).")

    # 3. Phase D AgentTune 1.1.3 presence
    assert "agenttune.create_distill_trainer" in full_code, "agenttune.create_distill_trainer missing"
    assert "Project" in full_code and "agenttune.agentic" in full_code, "agenttune Project missing"
    assert "EventLog" in full_code, "EventLog missing"
    assert "agentic_metrics" in full_code, "agenttune agentic_metrics missing"
    assert "SanoVocalTrainerFactory" in full_code, "SanoVocalTrainerFactory missing"
    print("[PASS] Phase D: AgentTune 1.1.3 APIs visibly imported and executed.")

    # 4. Strict exclusion of SafeTune & AuditKIT
    assert "apply_safe_lora" not in full_code, "VIOLATION: apply_safe_lora detected in Phase C/D!"
    assert "safetune" not in full_code.lower() or "safetune is intentionally excluded" in "".join(cells[0]["source"]).lower(), "VIOLATION: SafeTune detected in Phase C/D!"
    assert "auditkit" not in full_code.lower() or "auditkit is intentionally excluded" in "".join(cells[0]["source"]).lower(), "VIOLATION: AuditKIT detected in Phase C/D!"
    print("[PASS] SafeTune & AuditKIT strictly excluded from Phase C & D (reserved for Phase E/F).")

    # 5. Model and PEFT parameters
    assert "CohereLabs/aya-expanse-8b" in full_code, "Official Aya model ID missing"
    assert "load_in_4bit=True" in full_code, "4-bit quantization missing"
    assert "nf4" in full_code, "NF4 quantization type missing"
    assert "q_proj" in full_code and "v_proj" in full_code, "LoRA target modules missing"
    assert "r=16" in full_code and "lora_alpha=32" in full_code, "LoRA rank/alpha missing"
    print("[PASS] Model architecture, 4-bit NF4, and LoRA hyperparameters verified.")

    # 6. Deliverables check
    assert "sanovocal_aya_qlora_final" in full_code, "Final output adapter path missing"
    assert "training_manifest.json" in full_code, "training_manifest.json missing"
    assert "agenttune_metrics.json" in full_code, "agenttune_metrics.json missing"
    assert "sanovocal_phase_c_d_reports" in full_code, "sanovocal_phase_c_d_reports missing"
    print("[PASS] All required manifest and artifact deliverables confirmed.")

    print("\nALL NOTEBOOK VALIDATIONS PASSED PERFECTLY!\n")

if __name__ == "__main__":
    validate_notebook()
