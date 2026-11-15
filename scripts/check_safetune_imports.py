import sys
sys.stdout.reconfigure(encoding='utf-8')

# Check what's actually importable from safetune.interventions.recover
import safetune.interventions.recover as rec
print("Public names in safetune.interventions.recover:")
for name in sorted(dir(rec)):
    if not name.startswith('_'):
        print(f"  {name}")

print()
# Check safetune.core
try:
    import safetune.core.safereact as core_sr
    print("safetune.core.safereact public names:")
    for name in sorted(dir(core_sr)):
        if not name.startswith('_'):
            print(f"  {name}")
except Exception as e:
    print(f"Could not import safetune.core.safereact: {e}")

print()
# Verify which specific imports exist
imports_to_check = [
    ("safetune.interventions.recover", "apply_safereact"),
    ("safetune.interventions.recover", "apply_safe_lora"),
    ("safetune.interventions.recover", "SafeReActConfig"),
    ("safetune.core.safereact", "SafeReActConfig"),
    ("safetune.core.safereact", "apply_safereact"),
]
for mod_name, attr in imports_to_check:
    try:
        mod = __import__(mod_name, fromlist=[attr])
        obj = getattr(mod, attr, None)
        print(f"  {'OK' if obj else 'MISSING':8s}  from {mod_name} import {attr}")
    except Exception as e:
        print(f"  ERROR     from {mod_name} import {attr}  -> {e}")
