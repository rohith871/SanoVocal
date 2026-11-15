import json, sys
sys.stdout.reconfigure(encoding="utf-8")

p = r"C:\Users\KOYA ROHITH\.gemini\antigravity\brain\31ece8a2-b4ee-4cf3-9a3b-f867315457a2\.user_uploaded\media_1794567721037_63062540.ipynb"
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = "".join(cell.get("source", []))
    if any(k in src for k in ["phase_c_manifest", "agenttune_metrics", "baseline_evaluation_45_test", "adapter_config"]):
        print(f"Cell {idx}:")
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text = "".join(out.get("text", []))
                print(text[:400])
