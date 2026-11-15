#!/usr/bin/env python3
"""
SanoVocal Kaggle Notebook Push Script
======================================
Pushes the repaired notebook to Kaggle using the Kaggle CLI.

Prerequisites:
  1. Run: kaggle auth login  (OAuth browser login)
     OR place ~/.kaggle/kaggle.json with {"username":"...", "key":"..."}
     OR set env vars: KAGGLE_USERNAME + KAGGLE_KEY

  2. Run this script from the SanoVocal root directory:
     python scripts/push_to_kaggle.py

  3. After push completes, go to Kaggle → Your Notebooks → sanovocal-aya-qlora-training
     and click "Run All" (with T4 GPU accelerator + Internet enabled).
"""

import subprocess
import sys
import os
import json
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(ROOT, 'notebooks')
NOTEBOOK_PATH = os.path.join(NOTEBOOKS_DIR, 'sanovocal_aya_qlora_training.ipynb')
METADATA_PATH = os.path.join(NOTEBOOKS_DIR, 'kernel-metadata.json')


def check_kaggle_auth():
    """Verify Kaggle CLI is authenticated."""
    result = subprocess.run(
        ['kaggle', 'kernels', 'list', '--mine', '--page-size', '1'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("❌ Kaggle authentication required!")
        print()
        print("Option A (Recommended): OAuth login")
        print("  kaggle auth login")
        print()
        print("Option B: API key file")
        print("  1. Go to https://www.kaggle.com/settings/api")
        print("  2. Click 'Generate New Token'")
        print("  3. Save kaggle.json to: ~/.kaggle/kaggle.json")
        print()
        print("Option C: Environment variables")
        print("  set KAGGLE_USERNAME=your_username")
        print("  set KAGGLE_KEY=your_api_key")
        sys.exit(1)
    print("✓ Kaggle authentication verified.")


def verify_notebook():
    """Check the notebook file exists and has the right size."""
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"❌ Notebook not found: {NOTEBOOK_PATH}")
        sys.exit(1)
    
    nb_size = os.path.getsize(NOTEBOOK_PATH)
    print(f"✓ Notebook found: {nb_size:,} bytes")
    
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)
    
    cells = nb['cells']
    print(f"✓ Notebook cells: {len(cells)}")
    assert len(cells) == 11, f"Expected 11 cells, got {len(cells)}"
    
    # Verify critical keywords
    full_text = json.dumps(nb)
    for kw in ['agenttune', 'create_distill_trainer', 'EventLog', 'agentic_metrics',
                '667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3']:
        assert kw.lower() in full_text.lower(), f"Missing required keyword: {kw}"
    print("✓ All critical keywords verified in notebook.")


def push_notebook():
    """Push the notebook to Kaggle using the CLI."""
    print(f"\nPushing notebook from: {NOTEBOOKS_DIR}")
    
    result = subprocess.run(
        ['kaggle', 'kernels', 'push', '-p', NOTEBOOKS_DIR],
        capture_output=True, text=True, cwd=NOTEBOOKS_DIR
    )
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("\n✅ Notebook successfully pushed to Kaggle!")
        print()
        print("NEXT STEPS:")
        print("  1. Open https://www.kaggle.com/code (Your Notebooks)")
        print("  2. Find 'SanoVocal — Aya QLoRA Training (Phase C + D)'")
        print("  3. Confirm T4 GPU is selected as accelerator")
        print("  4. Confirm Internet is enabled")
        print("  5. Confirm the Aya Expanse 8B model is attached:")
        print("     cohereforai/aya-expanse/transformers/8b/1")
        print("  6. Add Kaggle Secret: HF_TOKEN = <your HuggingFace token>")
        print("  7. Click 'Run All' and monitor execution")
        print()
        print("EXPECTED OUTPUTS after successful run:")
        print("  /kaggle/working/evaluation_reports/baseline_evaluation_45_test.jsonl")
        print("  /kaggle/working/evaluation_reports/phase_c_baseline_manifest.json")
        print("  /kaggle/working/sanovocal_aya_qlora_final/adapter_model.safetensors")
        print("  /kaggle/working/sanovocal_aya_qlora_final/training_manifest.json")
        print("  /kaggle/working/sanovocal_aya_qlora_final.zip")
        print("  /kaggle/working/sanovocal_phase_c_d_reports.zip")
    else:
        print(f"\n❌ Push failed with return code {result.returncode}")
        sys.exit(1)


if __name__ == '__main__':
    print("=" * 60)
    print(" SANOVOCAL KAGGLE NOTEBOOK PUSH")
    print("=" * 60)
    check_kaggle_auth()
    verify_notebook()
    push_notebook()
