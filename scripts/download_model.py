import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
token = os.getenv("HF_token")
assert token, "HF_token missing in .env"

from huggingface_hub import snapshot_download

local_dir = Path("artifacts/model/base").resolve()
local_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"DOWNLOADING CohereLabs/tiny-aya-global to {local_dir}")
print("=" * 70)

# We use local_dir_use_symlinks=False so all files exist standalone in artifacts/model/base
# ignore heavy image assets if any, or download all repo files
downloaded_path = snapshot_download(
    repo_id="CohereLabs/tiny-aya-global",
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    token=token,
    ignore_patterns=["assets/*", ".eval_results/*"]
)

print(f"\n[DOWNLOAD COMPLETE] Saved at: {downloaded_path}")

# Verify downloaded files
required_files = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json"
]

missing = []
for req in required_files:
    p = local_dir / req
    if p.exists():
        size_mb = p.stat().st_size / (1024**2)
        print(f"  [OK] {req:30s} ({size_mb:.2f} MB)")
    else:
        missing.append(req)
        print(f"  [MISSING] {req:30s}")

# Check weights
shards = list(local_dir.glob("model-*.safetensors"))
print(f"\nModel weight shards found: {len(shards)}")
total_weight_bytes = sum(s.stat().st_size for s in shards)
print(f"Total weights size: {total_weight_bytes / (1024**3):.2f} GB")

if missing or len(shards) == 0:
    print("\n[ERROR] Model download incomplete!")
    sys.exit(1)
else:
    print("\n[VERIFIED] Local model cache verified for offline use.")
