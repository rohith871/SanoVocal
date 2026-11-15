import sys
sys.stdout.reconfigure(encoding='utf-8')
import inspect, importlib

print("=" * 70)
print("SAFETUNE SOURCE INSPECTION")
print("=" * 70)

import safetune
print(f"\nVersion: {safetune.__version__}")
print(f"File:    {safetune.__file__}")

print("\n" + "=" * 70)
print("apply_safereact — full source")
print("=" * 70)
from safetune.interventions.recover import apply_safereact
print(inspect.getsource(apply_safereact))

print("\n" + "=" * 70)
print("Searching for _run_safereact / core internals")
print("=" * 70)
try:
    from safetune.interventions.recover import safereact as sr_mod
    print(inspect.getsource(sr_mod))
except Exception as e:
    print(f"Could not import recover.safereact module: {e}")

try:
    import safetune.interventions.recover as rec_mod
    for name in dir(rec_mod):
        if not name.startswith('__'):
            obj = getattr(rec_mod, name)
            if callable(obj):
                try:
                    src = inspect.getsource(obj)
                    print(f"\n--- {name} ---")
                    print(src[:3000])
                except Exception:
                    pass
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 70)
print("train_lora=False path search")
print("=" * 70)
import safetune.interventions.recover as rm
full_src = inspect.getsource(rm)
# Find lines around train_lora
lines = full_src.splitlines()
for i, line in enumerate(lines):
    if 'train_lora' in line or 'weight' in line.lower() and ('merge' in line.lower() or 'copy' in line.lower()):
        start = max(0, i-2)
        end = min(len(lines), i+5)
        print(f"\n[Line ~{i}]")
        print('\n'.join(lines[start:end]))

print("\n" + "=" * 70)
print("reference_model CPU usage check")
print("=" * 70)
for i, line in enumerate(lines):
    if 'reference_model' in line or 'ref_sd' in line or 'state_dict' in line:
        start = max(0, i-1)
        end = min(len(lines), i+4)
        print(f"\n[Line ~{i}]")
        print('\n'.join(lines[start:end]))
