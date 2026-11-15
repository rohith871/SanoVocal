import inspect
from pathlib import Path
import safetune.core.neuron_ft as nft

out_file = Path("artifacts/safetune_neuron_ft.txt")
out_file.parent.mkdir(parents=True, exist_ok=True)

with open(out_file, "w", encoding="utf-8") as f:
    f.write("=== safetune.core.neuron_ft source ===\n")
    f.write(inspect.getsource(nft))

print(f"Wrote {out_file} ({out_file.stat().st_size} bytes)")
