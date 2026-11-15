import os
import sys
import time
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
token = os.getenv("HF_token")
assert token, "HF_token missing in .env"

REPO_ID = "CohereLabs/tiny-aya-global"
LOCAL_DIR = Path("artifacts/model/base").resolve()
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors"
]

def download_file(filename: str):
    target_path = LOCAL_DIR / filename
    part_path = LOCAL_DIR / f"{filename}.part"
    url = f"https://huggingface.co/{REPO_ID}/resolve/main/{filename}"
    
    # Check if already complete
    req_head = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="HEAD")
    try:
        with urllib.request.urlopen(req_head, timeout=30) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
    except Exception as e:
        print(f"Error checking HEAD for {filename}: {e}")
        total_size = 0

    if target_path.exists():
        actual_size = target_path.stat().st_size
        if total_size > 0 and actual_size == total_size:
            print(f"[EXISTS] {filename} ({actual_size / (1024**2):.2f} MB)")
            return
        elif total_size == 0 and actual_size > 0:
            print(f"[EXISTS] {filename} ({actual_size / (1024**2):.2f} MB)")
            return
        else:
            print(f"[INCOMPLETE] {filename} is {actual_size} bytes, expected {total_size}. Re-downloading...")

    # Resume support
    headers = {"Authorization": f"Bearer {token}"}
    existing_bytes = 0
    if part_path.exists():
        existing_bytes = part_path.stat().st_size
        if existing_bytes < total_size:
            headers["Range"] = f"bytes={existing_bytes}-"
            mode = "ab"
            print(f"[RESUME] {filename} from {existing_bytes / (1024**2):.2f} MB / {total_size / (1024**2):.2f} MB")
        else:
            existing_bytes = 0
            mode = "wb"
    else:
        mode = "wb"

    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    last_print = t0
    bytes_downloaded = existing_bytes

    print(f"[DOWNLOADING] {filename} (Total: {total_size / (1024**2):.2f} MB)...")
    with urllib.request.urlopen(req, timeout=60) as resp, open(part_path, mode) as f:
        chunk_size = 1024 * 1024 * 4  # 4 MB chunks
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            bytes_downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 5.0 or (total_size > 0 and bytes_downloaded >= total_size):
                speed = (bytes_downloaded - existing_bytes) / (now - t0 + 1e-5) / (1024**2)
                pct = (bytes_downloaded / total_size * 100) if total_size > 0 else 0
                print(f"  -> {filename}: {bytes_downloaded / (1024**2):.1f} / {total_size / (1024**2):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s")
                last_print = now

    # Rename part to final
    if part_path.exists():
        if target_path.exists():
            target_path.unlink()
        part_path.rename(target_path)
    print(f"[COMPLETED] {filename} ({target_path.stat().st_size / (1024**2):.2f} MB in {time.time() - t0:.1f}s)")

def main():
    print("=" * 70)
    print(f"STREAMING DIRECT DOWNLOAD: {REPO_ID} -> {LOCAL_DIR}")
    print("=" * 70)
    for f in FILES:
        download_file(f)
    print("\nAll files successfully verified and downloaded locally.")

if __name__ == "__main__":
    main()
