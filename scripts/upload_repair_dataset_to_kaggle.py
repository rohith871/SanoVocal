"""
Upload SanoVocal Tool Contract Repair Dataset to Kaggle.
Stages files from data/repair/ into artifacts/staging/sanovocal-tool-contract-repair/
and uploads via kagglehub.dataset_upload("koyarohith/sanovocal-tool-contract-repair", ...).
"""

import shutil
import json
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPAIR_DIR = ROOT / "data" / "repair"
STAGING_DIR = ROOT / "artifacts" / "staging" / "sanovocal-tool-contract-repair"
DATASET_HANDLE = "koyarohith/sanovocal-tool-contract-repair"

def refresh_kaggle_token():
    """Ensure active access token is present in ~/.kaggle/access_token."""
    try:
        tok = subprocess.check_output([
            str(ROOT / ".venv" / "Scripts" / "kaggle.exe"),
            "auth", "print-access-token"
        ]).decode().strip()
        p = Path(os.path.expanduser("~/.kaggle/access_token"))
        p.write_text(tok, encoding="utf-8")
        print(f"[OK] Refreshed Kaggle access token ({len(tok)} bytes).")
    except Exception as e:
        print(f"[WARN] Could not refresh token via CLI: {e}")

def main():
    print(f"Staging repair dataset for upload to Kaggle: {DATASET_HANDLE}...")
    refresh_kaggle_token()

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy required files
    files_to_copy = [
        "sanovocal_tool_contract_repair.jsonl",
        "repair_dataset_manifest.json",
        "repair_dataset_validation_report.md"
    ]

    for fname in files_to_copy:
        src = REPAIR_DIR / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing required repair file: {src}")
        dst = STAGING_DIR / fname
        shutil.copy2(src, dst)
        print(f"Copied {fname} ({dst.stat().st_size:,} bytes) to staging.")

    # 2. Add dataset-metadata.json
    metadata = {
        "title": "SanoVocal Tool Contract Repair Dataset",
        "id": DATASET_HANDLE,
        "licenses": [{"name": "CC0-1.0"}],
        "description": "High-precision tool-contract repair dataset for SanoVocal (226 examples) across 7 categories and 3 languages (English, Hindi, Hinglish), strictly verified via deterministic tool execution engine."
    }
    with open(STAGING_DIR / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Created dataset-metadata.json in staging.")

    # 3. Upload via kagglehub
    import kagglehub
    print(f"Calling kagglehub.dataset_upload('{DATASET_HANDLE}', '{STAGING_DIR}')...")
    kagglehub.dataset_upload(
        handle=DATASET_HANDLE,
        local_dataset_dir=str(STAGING_DIR),
        version_notes="Initial release of 226 verified tool-contract repair examples (Categories A-G)."
    )
    print("[SUCCESS] Dataset upload completed successfully!")

if __name__ == "__main__":
    main()
