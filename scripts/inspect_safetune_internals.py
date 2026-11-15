import inspect
from pathlib import Path
import safetune.core.safereact as sr

out_file = Path("artifacts/safetune_internals.txt")
out_file.parent.mkdir(parents=True, exist_ok=True)

with open(out_file, "w", encoding="utf-8") as f:
    f.write("=== safetune.core.safereact source ===\n")
    f.write(inspect.getsource(sr))

print(f"Wrote {out_file} ({out_file.stat().st_size} bytes)")
