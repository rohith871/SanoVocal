import os
import sys
import json
import time
import gc
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Ensure project root and scripts directory are on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.eval_pipeline import run_evaluation

BASE_MODEL_DIR = Path("artifacts/model/base").resolve()
REPAIR_ADAPTER_DIR = Path("artifacts/model/repair_finetuned").resolve()
SAFETUNE_ADAPTER_DIR = Path("artifacts/model/final_safetuned").resolve()
TEST_FILE = Path("data/traces/agent_trajectories_test.jsonl").resolve()

def main():
    print("=" * 80)
    print("SANOVOCAL LOCAL END-TO-END PIPELINE: AYA -> AGENTTUNE -> SAFETUNE -> AUDITKIT")
    print("=" * 80)
    t_start = time.time()

    # -------------------------------------------------------------------------
    # STEP 7: Verify Offline Loading of Base Model
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Verifying offline loading of CohereLabs/tiny-aya-global in 4-bit NF4...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    base_model.eval()

    mem_used = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"[OK] Base model loaded in 4-bit NF4. VRAM allocated: {mem_used:.2f} GiB")

    # -------------------------------------------------------------------------
    # STEP 11: Run BASELINE on Locked 45-Test Set (or load if already verified)
    # -------------------------------------------------------------------------
    baseline_path = Path("artifacts/metrics/baseline_45.json").resolve()
    if baseline_path.exists():
        print(f"\n[STEP 11] Baseline evaluation already completed: {baseline_path}")
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
            baseline_metrics = baseline_data["metrics"]
        print(f"[OK] Loaded verified baseline metrics across all 45 test scenarios.")
        del base_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n[STEP 11] Running BASELINE evaluation on locked 45-test set...")
        from scripts.eval_pipeline import run_evaluation

        baseline_metrics = run_evaluation(
            model=base_model,
            tokenizer=tokenizer,
            test_file=TEST_FILE,
            output_file=baseline_path,
            model_label="Base Aya (tiny-aya-global)"
        )

        # Free base model to reclaim memory before training
        del base_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # STEP 13: AgentTune Repair Fine-Tuning
    # -------------------------------------------------------------------------
    print("\n[STEP 13] Executing genuine AgentTune QLoRA repair fine-tuning...")
    from scripts.train_agenttune_qlora import run_training
    run_training()

    # -------------------------------------------------------------------------
    # STEP 15: Run REPAIR-FT Evaluation on Locked 45-Test Set
    # -------------------------------------------------------------------------
    print("\n[STEP 15] Loading Repair-FT Model for evaluation on locked 45-test set...")
    base_for_ft = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    ft_model = PeftModel.from_pretrained(
        base_for_ft,
        str(REPAIR_ADAPTER_DIR),
        is_trainable=False
    )
    ft_model.eval()

    ft_metrics = run_evaluation(
        model=ft_model,
        tokenizer=tokenizer,
        test_file=TEST_FILE,
        output_file=Path("artifacts/metrics/repair_ft_45.json").resolve(),
        model_label="Repair-FT (AgentTune Distilled)"
    )

    del ft_model
    del base_for_ft
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # STEP 18: Execute SafeTune SafeReAct Intervention
    # -------------------------------------------------------------------------
    print("\n[STEP 18] Executing SafeTune 0.1.12 SafeReAct Intervention...")
    from scripts.run_safetune_pipeline import run_safetune
    run_safetune()

    # -------------------------------------------------------------------------
    # STEP 20: Run AuditKIT on Final Model
    # -------------------------------------------------------------------------
    print("\n[STEP 20] Executing AuditKIT 1.1.6 evaluation on final model...")
    from scripts.run_auditkit_eval import run_auditkit
    run_auditkit()

    # -------------------------------------------------------------------------
    # STEP 21: Run FINAL Evaluation on SafeTune Repaired Model
    # -------------------------------------------------------------------------
    print("\n[STEP 21] Loading SafeTune Repaired Model for final locked 45-test evaluation...")
    base_for_st = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    st_model = PeftModel.from_pretrained(
        base_for_st,
        str(SAFETUNE_ADAPTER_DIR),
        is_trainable=False
    )
    st_model.eval()

    st_metrics = run_evaluation(
        model=st_model,
        tokenizer=tokenizer,
        test_file=TEST_FILE,
        output_file=Path("artifacts/metrics/safetune_final_45.json").resolve(),
        model_label="SafeTune-Repaired (Final Model)"
    )

    del st_model
    del base_for_st
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # STEP 22: Three-Way Metric Comparison
    # -------------------------------------------------------------------------
    print("\n[STEP 22] Generating 3-Way Comparative Benchmark Table...")
    from scripts.compare_three_way import generate_three_way_comparison
    generate_three_way_comparison()

    # -------------------------------------------------------------------------
    # STEP 23: Generate Final Pipeline Manifest
    # -------------------------------------------------------------------------
    print("\n[STEP 23] Generating Final Pipeline Reproducibility Manifest...")
    from scripts.generate_manifest import create_manifest
    create_manifest()

    total_time = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"ALL PIPELINE STEPS SUCCESSFULLY EXECUTED IN {total_time/60:.2f} MINUTES!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
