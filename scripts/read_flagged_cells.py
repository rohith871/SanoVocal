import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = 'notebooks/sanovocal_phase_d_safetune_auditkit.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print Cell 18 full source
cell = nb['cells'][18]
src = ''.join(cell.get('source', []))
print("=== CELL 18 FULL SOURCE ===")
print(src)

# Also show cell 19 (comparative table)
print("\n=== CELL 19 FULL SOURCE ===")
cell19 = nb['cells'][19]
src19 = ''.join(cell19.get('source', []))
print(src19)

# Cell 20 (final report)
print("\n=== CELL 20 FULL SOURCE ===")
cell20 = nb['cells'][20]
src20 = ''.join(cell20.get('source', []))
print(src20[:3000])
