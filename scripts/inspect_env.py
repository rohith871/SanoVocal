import sys
import os
import torch

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("SANOVOCAL LOCAL ENVIRONMENT INSPECTION")
print("=" * 60)
print(f"Python: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
    print(f"Total VRAM: {total_mem:.2f} GiB")
    print(f"Free VRAM: {free_mem:.2f} GiB")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

print("\nInstalled Packages Inspection:")
packages = [
    'transformers', 'peft', 'trl', 'bitsandbytes', 'accelerate', 'datasets',
    'agenttune', 'safetune', 'auditkit'
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        loc = getattr(mod, '__file__', 'unknown')
        print(f"  [OK] {pkg:15s} v{ver:<10s} -> {loc}")
    except Exception as e:
        print(f"  [FAIL] {pkg:15s} ERROR: {e}")

print("=" * 60)
