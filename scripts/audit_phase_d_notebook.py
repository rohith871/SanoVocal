import json, sys
sys.stdout.reconfigure(encoding='utf-8')

HARDCODED_TOKENS = [
    '66.7', '91.3', '88.9', '95.6', '86.7', '80.0', '94.2', '97.8',
    '18.2', '28.5', '35.5', '24.0', '100.0',
    # broader patterns
    'BASE_AYA_SCORE', 'PHASE_C_SCORE', 'REPAIRED_SCORE',
]

nb_path = 'notebooks/sanovocal_phase_d_safetune_auditkit.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f'Total cells: {len(cells)}\n')
print(f"{'Idx':<4} {'Type':<10} {'Status':<12} First line / flags")
print("-" * 100)

flagged = []
for i, cell in enumerate(cells):
    src = ''.join(cell.get('source', []))
    first_line = src.splitlines()[0].strip() if src.strip() else '<empty>'
    ct = cell['cell_type']
    hits = [t for t in HARDCODED_TOKENS if t in src]
    status = 'HARDCODED' if hits else 'OK'
    if hits:
        flagged.append((i, ct, hits, src))
    print(f"{i:<4} {ct:<10} {status:<12} {first_line[:75]}")

print(f"\n\n{'=' * 70}")
print(f"FLAGGED CELLS: {len(flagged)}")
print('=' * 70)
for idx, ct, hits, src in flagged:
    print(f"\n--- Cell {idx} ({ct}) ---")
    print(f"Tokens found: {hits}")
    # Show only the lines containing the tokens
    for line in src.splitlines():
        if any(t in line for t in hits):
            print(f"  >> {line.rstrip()}")
