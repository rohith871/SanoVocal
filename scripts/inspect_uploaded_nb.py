import json

p = r"C:\Users\KOYA ROHITH\.gemini\antigravity\brain\31ece8a2-b4ee-4cf3-9a3b-f867315457a2\.user_uploaded\media_1794567721037_63062540.ipynb"
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    first_line = cell["source"][0].strip() if cell.get("source") else "<empty>"
    if "cell 50" in src.lower() or "repair" in src.lower() or i >= 40:
        print(f"Index {i}: {cell.get('cell_type')} | First line: {first_line[:80]}")
